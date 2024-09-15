import streamlit as st
import pandas as pd
import numpy as np
import xlsxwriter
import openpyxl
from st_aggrid import AgGrid
import io
from scipy.integrate import quad
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
import plotly.express as px

# Set the app title and description
st.set_page_config(
    page_title="PSI and MPSI-MARA calculator",
    #page_icon=":chart_with_upwards_trend:",  # You can customize the icon
    #layout="wide",  # You can set the layout (wide or center)
    initial_sidebar_state="auto"  # You can set the initial sidebar state
)

def download_template():
    # Adjust based on the number of alternatives and criteria
    num_alternatives = 9  # You can set a default number or ask the user for input
    num_criteria = 17  # Same for criteria

    # Generate a list of alternative names
    alternatives = [f'A{i+1}' for i in range(num_alternatives)]

    # Create data for the template: "C1", "C2", ..., in the first row, and "Max/Min" in the second row
    criteria_labels = [f'C{i+1}' for i in range(num_criteria)]
    benefit_cost_row = ['Max' if i < 10 else 'Min' for i in range(num_criteria)]  # First 10 are Max, rest are Min

    # Prepare data for the DataFrame
    data = {f'C{i+1}': [''] * num_alternatives for i in range(num_criteria)}
    df = pd.DataFrame(data)

    # Set the first row for the "C1", "C2", ..., and second row for the "Max/Min"
    df.loc[-2] = criteria_labels
    df.loc[-1] = benefit_cost_row
    df.index = df.index + 2  # Shifting the index to make space for the new rows
    df = df.sort_index()

    # Add the "A/C" column for alternatives
    df.insert(0, 'A/C', ['A/C'] + [''] + alternatives)

    # Convert the DataFrame to an Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, header=False)

    # Provide a download link for the template
    st.download_button(
        label="Download Excel template",
        data=excel_buffer,
        file_name="MEREC_SPOTIS_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Function to read Excel file
def read_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # Show the first two rows for debugging purposes
    #st.write("First Row (Headers):")
    #st.write(df.columns.tolist())
    
    #st.write("Second Row (Criterion Types):")
    #st.write(df.iloc[0, 1:].tolist())

    # Extract the "Max" or "Min" labels from the second row (the row defining if it's Benefit or Cost)
    criterion_types = df.iloc[0, 1:].apply(lambda x: 'Benefit' if str(x).strip().lower() == 'max' else 'Cost').tolist()

    # Remove the second row (the row with "Max" or "Min" labels) from the DataFrame
    df = df.drop(0).reset_index(drop=True)

    # Rename columns to C1, C2, etc. and keep the first column as 'A/C'
    num_criteria = len(df.columns) - 1
    columns = ['A/C'] + [f'C{i+1}' for i in range(num_criteria)]
    df.columns = columns

    # Convert all the criteria columns (except the 'A/C' column) to numeric values
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, criterion_types, df.shape[0], num_criteria

def get_payoff_matrix():
    num_alternatives = st.number_input("Enter the number of alternatives:", min_value=2, value=2, step=1)
    num_criteria = st.number_input("Enter the number of criteria:", min_value=1, value=1, step=1)

    # Create a DataFrame to hold the payoff matrix
    columns = ['A/C'] + [f'C{i+1}' for i in range(num_criteria)]
    data = [[f'A{j+1}'] + [0 for _ in range(num_criteria)] for j in range(num_alternatives)]
    payoff_matrix = pd.DataFrame(data, columns=columns)

    # Create an ag-Grid component
    grid_response = AgGrid(payoff_matrix, editable=True, index=False, fit_columns_on_grid_load=True)

    # Get the edited DataFrame from the AgGrid response
    edited_matrix = grid_response['data']

    # Get the type of each criterion (Benefit or Cost)
    criterion_types = []
    for i in range(num_criteria):
        criterion_label = f"C{i+1}"
        criterion_type = st.selectbox(f"{criterion_label} - Benefit or Cost?", ["Benefit", "Cost"])
        criterion_types.append(criterion_type)

    return edited_matrix, criterion_types

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

def calculate_phi_j(p_values):
    phi_values = 1 - p_values
    return phi_values

def calculate_psi_j(phi_values):
    psi_values = phi_values / phi_values.sum()
    return psi_values

def calculate_w_ij(p_values):
    w_values = p_values / p_values.sum()
    return w_values

def calculate_variables(normalized_df):
    v_values = calculate_v_ij(normalized_df)
    p_values = calculate_p_ij(normalized_df, v_values)
    w_values = calculate_w_ij(p_values)
    variables_df = pd.DataFrame({'v': v_values, 'p': p_values, 'w': w_values})
    return variables_df

def calculate_PSI_variables(normalized_df):
    v_values = calculate_v_ij(normalized_df)
    p_values = calculate_p_ij(normalized_df, v_values)
    phi_values = calculate_phi_j(p_values)
    psi_values = calculate_psi_j(phi_values)
    PSI_variables_df = pd.DataFrame({'phi': phi_values, 'psi': psi_values})
    return PSI_variables_df

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

    # Create a PDF document in memory using BytesIO
    buffer = io.BytesIO()
    
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

    # Save the generated PDF in a BytesIO object
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(elements)

    # Reset the buffer position to the beginning
    buffer.seek(0)

    # # Offer the PDF file for download with a download button
    # st.download_button("Download PDF Report", data=buffer, file_name="mcda_report.pdf", mime="application/pdf")

    return buffer

def main():
    menu = ["Home", "PSI", "MPSI-MARA", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Home")
        st.subheader("PSI and MPSI-MARA Calculator")
        st.write("This is a MCDA Calculator for the PSI and MPSI-MARA Methods.")
        st.write("To use this Calculator:")
        st.write("1. Define how many alternatives and criteria you'll measure.")
        st.write("2. Define if the criteria are of benefit (more is better) or cost (less is better).")

    elif choice == "MPSI-MARA":
        st.title("MPSI-MARA Hybrid Method MCDA Calculator")

        # Input Method
        data_input_method = st.selectbox("Select Data Input Method:", ["Manual Input", "Upload Excel"])

        if data_input_method == "Upload Excel":
            st.write("Download the template to fill out the data:")
            download_template()
            uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

            if uploaded_file:
                payoff_matrix, criterion_types, _, num_criteria = read_excel(uploaded_file)
                st.subheader("Payoff Matrix from Uploaded Excel:")
                st.dataframe(payoff_matrix)

        elif data_input_method == "Manual Input":
            payoff_matrix, criterion_types, _, num_criteria = get_payoff_matrix()
            st.subheader("Payoff Matrix:")
            st.dataframe(payoff_matrix)

        if 'payoff_matrix' in locals():
            normalized_matrix = normalize_matrix(payoff_matrix, criterion_types)
            st.subheader("Normalized Matrix:")
            st.dataframe(normalized_matrix)

            # Further calculations for MPSI-MARA
            variables_df = calculate_variables(normalized_matrix)
            st.subheader("Calculated Variables:")
            st.dataframe(variables_df)

            new_matrix = calculate_new_matrix(normalized_matrix, variables_df['w'])
            st.subheader("New Matrix:")
            st.dataframe(new_matrix)

            set_Sj = create_set_Sj(new_matrix)
            set_Smax, set_Smin = split_sets_Smax_Smin(criterion_types, set_Sj)

            st.subheader("Set S_j:")
            st.dataframe(pd.DataFrame(list(set_Sj.items()), columns=["Criterion", "Value"]).T)
            st.subheader("Set S_max:")
            st.dataframe(pd.DataFrame(list(set_Smax.items()), columns=["Criterion", "Value"]).T)
            st.subheader("Set S_min:")
            st.dataframe(pd.DataFrame(list(set_Smin.items()), columns=["Criterion", "Value"]).T)

            set_Tmax, set_Tmin = create_set_Tmax_Tmin(new_matrix, criterion_types)
            st.subheader("Set T_i^max:")
            st.dataframe(pd.DataFrame(list(set_Tmax.items()), columns=["Alternative", "T_max"]).T)
            st.subheader("Set T_i^min:")
            st.dataframe(pd.DataFrame(list(set_Tmin.items()), columns=["Alternative", "T_min"]).T)

            T_ik, T_il = calculate_T_ik_T_il(set_Tmax, set_Tmin)
            st.subheader("T_ik for each alternative:")
            st.dataframe(pd.DataFrame(list(T_ik.items()), columns=["Alternative", "T_ik"]).T)
            st.subheader("T_il for each alternative:")
            st.dataframe(pd.DataFrame(list(T_il.items()), columns=["Alternative", "T_il"]).T)

            # Calculate the optimal alternative function
            Sk = sum(set_Smax.values())
            Sl = sum(set_Smin.values())
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
                pdf_file = generate_pdf_report(payoff_matrix, normalized_matrix, variables_df, new_matrix,
                                               set_Sj, set_Smax, set_Smin, set_Tmax, set_Tmin, T_ik, T_il,
                                               def_opt_integral, alternative_functions, def_integrals, ranked_alternatives,
                                               Sk, Sl)
                st.download_button("Download PDF", data=pdf_file, file_name="mcda_report.pdf", mime="application/pdf")
                st.success("PDF report generated successfully!")

    elif choice == "PSI":
        st.title("PSI Calculator")

        # Input Method
        data_input_method = st.selectbox("Select Data Input Method:", ["Manual Input", "Upload Excel"])

        if data_input_method == "Upload Excel":
            st.write("Download the template to fill out the data:")
            download_template()
            uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

            if uploaded_file:
                payoff_matrix, criterion_types, _, num_criteria = read_excel(uploaded_file)
                st.subheader("Payoff Matrix from Uploaded Excel:")
                st.dataframe(payoff_matrix)

        elif data_input_method == "Manual Input":
            payoff_matrix, criterion_types, _, num_criteria = get_payoff_matrix()
            st.subheader("Payoff Matrix:")
            st.dataframe(payoff_matrix)

        if 'payoff_matrix' in locals():
            normalized_matrix = normalize_matrix(payoff_matrix, criterion_types)
            st.subheader("Normalized Matrix:")
            st.dataframe(normalized_matrix)

            PSI_variables_df = calculate_PSI_variables(normalized_matrix)
            st.subheader("Calculated Variables:")
            st.dataframe(PSI_variables_df)

            # Plot the PSI weights using Plotly bar chart
            fig = px.bar(
                PSI_variables_df,
                x=PSI_variables_df.index,  # Assuming your criteria have meaningful names
                y='psi',
                labels={'index': 'Criteria', 'psi': 'PSI Weight'},
                title='PSI Weights for Criteria',
            )
            fig.update_layout(xaxis_title_text='Criteria', yaxis_title_text='PSI Weight', xaxis_tickangle=-45)
            st.plotly_chart(fig)

    else:
        st.subheader("About")
        st.write("The PSI Method is a method created by Maniya et al. [2010]")
        st.write("The Hybrid MCDA Method MPSI-MARA is a method created by Gligoric et al. [2022]")
        st.write("Both Articles")
        st.write("https://www.sciencedirect.com/science/article/abs/pii/S0261306909006396?via%3Dihub")
        st.write('https://www.mdpi.com/2079-8954/10/6/248')
        st.write("To cite this work:")
        st.write("Araujo, Tullio Mozart Pires de Castro; Gomes, Carlos Francisco Simões.; Santos, Marcos dos. PSI and MPSI-MARA For Decision Making (v1), Universidade Federal Fluminense, Niterói, Rio de Janeiro, 2023.")
    
    # Add logo to the sidebar
    logo_path = "https://i.imgur.com/g7fITf4.png"  # Replace with the actual path to your logo image file
    st.sidebar.image(logo_path, use_column_width=True)


if __name__ == "__main__":
    main()
