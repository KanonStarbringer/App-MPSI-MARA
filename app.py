import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.integrate import quad
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle


def get_payoff_matrix():
    num_alternatives = st.number_input("Enter the number of alternatives:", min_value=2, value=2, step=1)
    num_criteria = st.number_input("Enter the number of criteria:", min_value=1, value=1, step=1)

    # Create a DataFrame to hold the payoff matrix
    columns = ['A/C'] + [f'C{i+1}' for i in range(num_criteria)]
    data = [[f'A{j+1}'] + [0 for _ in range(num_criteria)] for j in range(num_alternatives)]
    payoff_matrix = pd.DataFrame(data, columns=columns)

    # Get the type of each criterion (Benefit or Cost)
    criterion_types = []
    for i in range(num_criteria):
        criterion_label = f"C{i+1}"
        criterion_type = st.selectbox(f"{criterion_label} - Benefit or Cost?", ["Benefit", "Cost"])
        criterion_types.append(criterion_type)

    # Input the values for the payoff matrix using DataFrame
    st.subheader("Enter the values for the payoff matrix:")
    input_matrix = pd.DataFrame(data, columns=columns)
    for i in range(num_alternatives):
        for j in range(num_criteria):
            input_matrix.iloc[i, j + 1] = st.number_input(f"{columns[j+1]} - {data[i][0]}", value=0.0, step=0.01)

    payoff_matrix.iloc[:, 1:] = input_matrix.iloc[:, 1:]

    return payoff_matrix, criterion_types

def normalize_matrix(df, criterion_types):
    normalized_df = df.copy()
    for j, criterion_type in enumerate(criterion_types):
        if criterion_type == "Benefit":
            col_max = df.iloc[:, j+1].max()
            normalized_df.iloc[:, j+1] = df.iloc[:, j+1] / col_max
        else:
            col_min = df.iloc[:, j+1].min()
            normalized_df.iloc[:, j+1] = col_min / df.iloc[:, j+1]
    return normalized_df

def calculate_v_ij(normalized_df):
    v_values = normalized_df.iloc[:, 1:].mean()
    return v_values

def calculate_p_ij(normalized_df, v_values):
    p_values = ((normalized_df.iloc[:, 1:] - v_values) ** 2).sum()
    return p_values

def calculate_w_ij(p_values):
    w_values = p_values / p_values.sum()
    return w_values

def calculate_variables(normalized_df):
    v_values = calculate_v_ij(normalized_df)
    p_values = calculate_p_ij(normalized_df, v_values)
    w_values = calculate_w_ij(p_values)
    variables_df = pd.DataFrame({'v': v_values, 'p': p_values, 'w': w_values})
    return variables_df

def calculate_new_matrix(normalized_df, w_values):
    new_matrix = normalized_df.copy()
    new_matrix.iloc[:, 1:] = new_matrix.iloc[:, 1:] * w_values.values
    return new_matrix

def create_set_Sj(normalized_df):
    set_Sj = {}
    for col in normalized_df.columns[1:]:
        set_Sj[col] = normalized_df[col].max()
    return set_Sj

def split_sets_Smax_Smin(criterion_types, set_Sj):
    set_Smax = {}
    set_Smin = {}
    for col, val in set_Sj.items():
        if criterion_types[int(col[1:]) - 1] == "Benefit":
            set_Smax[col] = val
        else:
            set_Smin[col] = val
    return set_Smax, set_Smin

def create_set_Tmax_Tmin(new_matrix, criterion_types):
    set_Tmax = {}
    set_Tmin = {}
    for i, alternative in enumerate(new_matrix['A/C']):
        T_max = []
        T_min = []
        for j, criterion_type in enumerate(criterion_types):
            col = f"C{j+1}"
            if criterion_type == "Benefit":
                T_max.append(new_matrix[col].iloc[i])
            else:
                T_min.append(new_matrix[col].iloc[i])
        set_Tmax[alternative] = T_max
        set_Tmin[alternative] = T_min
    return set_Tmax, set_Tmin

def calculate_T_ik_T_il(set_Tmax, set_Tmin):
    T_ik = {}
    T_il = {}
    for alternative, Tmax in set_Tmax.items():
        T_ik[alternative] = sum(Tmax)
    for alternative, Tmin in set_Tmin.items():
        T_il[alternative] = sum(Tmin)
    return T_ik, T_il

def optimal_alternative_function(Sk, Sl):
    def f_opt(x):
        return (Sl - Sk) * x + Sk
    return f_opt

def alternative_function(T_ik, T_il):
    def f_i(x):
        return (T_il - T_ik) * x + T_ik
    return f_i

def calculate_definite_integral(func, a, b):
    integral_value, _ = quad(func, a, b)
    return integral_value


def generate_pdf_report(payoff_matrix, normalized_matrix, variables_df, new_matrix,
                        set_Sj, set_Smax, set_Smin, set_Tmax, set_Tmin, T_ik, T_il,
                        def_opt_integral, alternative_functions, def_integrals, ranked_alternatives,
                        Sk, Sl):

    # Create a new PDF document using SimpleDocTemplate with A4 paper size
    doc = SimpleDocTemplate("mcda_report.pdf", pagesize=A4)

    # Define styles for the report
    styles = getSampleStyleSheet()

    # Add the content to the PDF using a list of flowables
    elements = []

    # Add the title and other content to elements list using Paragraph
    title_text = "MPSI-MARA Hybrid Method MCDA Report"
    elements.append(Paragraph(title_text, styles['Title']))

    # Add the payoff matrix as a table to the PDF
    payoff_table_data = [['A/C'] + list(payoff_matrix.columns[1:])] + payoff_matrix.values.tolist()
    payoff_table = Table(payoff_table_data)
    # Apply TableStyle to the table for better formatting (optional)
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
    payoff_table.setStyle(style)
    elements.append(payoff_table)

    # Add the optimal alternative function, alternative functions, definite integrals, and ranking
    # to the PDF using Paragraph

    elements.append(Paragraph("Optimal Alternative Function:", styles['Heading2']))
    elements.append(Paragraph(f"f_opt(x) = ({Sl} - {Sk}) * x + {Sk}", styles['Normal']))

    elements.append(Paragraph("Alternative Functions:", styles['Heading2']))
    for alternative, f_i in alternative_functions.items():
        elements.append(Paragraph(f"f_{alternative}(x) = ({T_il[alternative]} - {T_ik[alternative]}) * x + {T_ik[alternative]}", styles['Normal']))

    elements.append(Paragraph("Definite Integrals of Alternative Functions:", styles['Heading2']))
    for alternative, def_i_integral in def_integrals.items():
        elements.append(Paragraph(f"Definite Integral of f_{alternative}(x): {def_i_integral}", styles['Normal']))

    elements.append(Paragraph(f"Definite Integral of Optimal Alternative Function: {def_opt_integral}", styles['Heading2']))

    elements.append(Paragraph("Ranking of Alternatives:", styles['Heading2']))
    for rank, (alternative, difference) in enumerate(ranked_alternatives, start=1):
        elements.append(Paragraph(f"Rank {rank}: Alternative {alternative}, Difference: {difference:.4f}", styles['Normal']))

    # Build the PDF document with the elements
    doc.build(elements)
    
    # Save the generated PDF in a BytesIO object
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(elements)

    # Reset the buffer position to the beginning
    buffer.seek(0)

    # Offer the PDF file for download with a download button
    st.download_button("Download PDF Report", data=buffer, file_name="mcda_report.pdf", mime="application/pdf")

def main():
    menu = ["Home", "Method", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Home")
        st.subheader("MPSI-MARA Calculator")
        st.write("This is a MCDA Calculator for the MPSI-MARA Method")
        st.write("To use this Calculator, is quite intuitive:")
        st.write("First, define how many alternatives and criteria you'll measure")
        st.write("Then, define if the criteria are of benefit (more is better)")
        st.write("Or, if the criteria are of cost (if less is better)")

    elif choice == "Method":
        st.title("MPSI-MARA Hybrid Method MCDA Calculator")

        payoff_matrix, criterion_types = get_payoff_matrix()
        st.subheader("Payoff Matrix:")
        st.dataframe(payoff_matrix)

        normalized_matrix = normalize_matrix(payoff_matrix, criterion_types)
        st.subheader("Normalized Matrix:")
        st.dataframe(normalized_matrix)

        variables_df = calculate_variables(normalized_matrix)
        st.subheader("Calculated Variables:")
        st.dataframe(variables_df)

        new_matrix = calculate_new_matrix(normalized_matrix, variables_df['w'])
        st.subheader("New Matrix:")
        st.dataframe(new_matrix)

        set_Sj = create_set_Sj(new_matrix)
        set_Smax, set_Smin = split_sets_Smax_Smin(criterion_types, set_Sj)

        st.subheader("Set S_j:")
        st.write(set_Sj)
        st.subheader("Set S_max:")
        st.write(set_Smax)
        st.subheader("Set S_min:")
        st.write(set_Smin)

        set_Tmax, set_Tmin = create_set_Tmax_Tmin(new_matrix, criterion_types)

        st.subheader("Set T_i^max:")
        st.write(set_Tmax)
        st.subheader("Set T_i^min:")
        st.write(set_Tmin)

        T_ik, T_il = calculate_T_ik_T_il(set_Tmax, set_Tmin)

        st.subheader("T_ik for each alternative:")
        st.write(T_ik)
        st.subheader("T_il for each alternative:")
        st.write(T_il)

        # Calculate the optimal alternative function
        Sk = sum(set_Smax.values())
        st.subheader("Sk")
        st.write(Sk)
        Sl = sum(set_Smin.values())
        st.subheader("Sl")
        st.write(Sl)
        f_opt = optimal_alternative_function(Sk, Sl)

        st.subheader("Optimal Alternative Function:")
        st.write(f"f_opt(x) = ({Sl} - {Sk}) * x + {Sk}")


        # Calculate the alternative functions for each alternative
        alternative_functions = {}
        for alternative in T_ik.keys():
            f_i = alternative_function(T_ik[alternative], T_il[alternative])
            alternative_functions[alternative] = f_i

        st.subheader("Alternative Functions:")
        for alternative, f_i in alternative_functions.items():
            st.write(f"f_{alternative}(x) = ({T_il[alternative]} - {T_ik[alternative]}) * x + {T_ik[alternative]}")

    # Calculate the definite integral of the Optimal Alternative Function
        def_opt_integral = calculate_definite_integral(f_opt, 0, 1)
        st.subheader("Definite Integral of Optimal Alternative Function:")
        st.write(def_opt_integral)

        # Calculate the definite integrals of the Alternative Functions for each alternative
        st.subheader("Definite Integrals of Alternative Functions:")
        def_integrals = {}  # Dictionary to store the definite integrals for each alternative
        for alternative, f_i in alternative_functions.items():
            def_i_integral = calculate_definite_integral(f_i, 0, 1)
            st.write(f"Definite Integral of f_{alternative}(x):")
            st.write(def_i_integral)
            def_integrals[alternative] = def_i_integral

        # Calculate the differences and rank the alternatives
        ranked_alternatives = sorted(def_integrals, key=lambda alternative: def_opt_integral - def_integrals[alternative])

        # Display the ranking
        st.subheader("Ranking of Alternatives:")
        for rank, alternative in enumerate(ranked_alternatives, start=1):
            st.write(f"Rank {rank}: Alternative {alternative}")
        ranked_alternatives = []
        for alternative, def_i_integral in def_integrals.items():
            difference = def_opt_integral - def_i_integral
            ranked_alternatives.append((alternative, difference))

        ranked_alternatives = sorted(ranked_alternatives, key=lambda x: x[1])

        # Display the ranking with difference values
        st.subheader("Ranking of Alternatives:")
        for rank, (alternative, difference) in enumerate(ranked_alternatives, start=1):
            st.write(f"Rank {rank}: Alternative {alternative}, Difference: {difference:.4f}")

        if st.button("Generate PDF"):
            generate_pdf_report(payoff_matrix, normalized_matrix, variables_df, new_matrix,
                                set_Sj, set_Smax, set_Smin, set_Tmax, set_Tmin, T_ik, T_il,
                                def_opt_integral, alternative_functions, def_integrals, ranked_alternatives,
                                Sk, Sl)  # Pass Sk and Sl as arguments here
            st.success("PDF report generated successfully!")

    else:
        st.subheader("About")
        st.write("The Hybrid MCDA Method MPSI-MARA is method created by Gligoric et al. [2022]")
        st.write("Original Article")
        st.write('https://www.mdpi.com/2079-8954/10/6/248')
    
    # Add logo to the sidebar
    logo_path = "https://imgur.com/g7fITf4"  # Replace with the actual path to your logo image file
    st.sidebar.image(logo_path, use_column_width=True)


if __name__ == "__main__":
    main()
