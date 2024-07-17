import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook

class Extract:
    def __init__(self, file_path):
        self.file_path = file_path
        self.variables = []
        self.var_to_type = {}
        self.adjacency_matrix = None
        self.interactions_matrix = None
        self.test_extraction()  # Call the test_extraction function when the class is loaded


    def extract_adjacency_matrix(self):
        """Extract the adjacency matrix from an Excel table exported from Kumu (Kumu.io).
        The Kumu excel file contains one sheet with the CLD's variables ('Elements').
        It also contains a sheet with the CLD's causal links ('Connections').
        If there are known interactions in the system, these can be added in the 'Interactions' sheet.
        """
        # Read the elements, connections, and interactions sheets in the Kumu Excel file
        df_e = pd.read_excel(self.file_path, sheet_name="Elements")
        df_c = pd.read_excel(self.file_path, sheet_name="Connections")

        # Extract relevant columns
        df_e = df_e[["Label", "Type"]]   
        df_c = df_c[["From", "Type", "To"]]

        # Extract variables from the Elements 
        self.variables = list(df_e["Label"])
        self.var_to_type = dict(zip(list(df_e["Label"]), list(df_e["Type"])))

        # Create an empty adjacency matrix
        num_variables = len(self.variables)
        self.adjacency_matrix = np.zeros((num_variables, num_variables))

        # Populate the adjacency matrix
        for i, origin in enumerate(df_c["From"]):
            destination = df_c["To"][i]

            # Determine the polarity
            polarity = 0
            temp = df_c["Type"][i]
            if str(temp) == '+':
                polarity = 1
            elif str(temp) == '-':
                polarity = -1

            # Calculate indices
            origin_index = self.variables.index(origin)
            destination_index = self.variables.index(destination)

            # Add polarity to adjacency matrix
            self.adjacency_matrix[destination_index, origin_index] = polarity

    def extract_interactions_matrix(self):
        """Extract the interactions matrix from the 'Interactions' sheet in the Kumu Excel file."""
        wb = load_workbook(self.file_path, read_only=True)   # open an Excel file and return a workbook

        if 'Interactions' in wb.sheetnames:
            df_i = pd.read_excel(self.file_path, sheet_name="Interactions")
            df_i = df_i[["From1", "From2", "Type", "To"]]

            # Create an empty matrix to annotate interactions
            num_variables = len(self.variables)
            self.interactions_matrix = np.zeros((num_variables, num_variables, num_variables))

            # Populate the interactions matrix
            for i, origin_1 in enumerate(df_i["From1"]):
                origin_2 = df_i["From2"][i]
                destination = df_i["To"][i]

                # Determine the polarity
                polarity = 0
                temp = df_i["Type"][i]
                if str(temp) == '+':
                    polarity = 1
                elif str(temp) == '-':
                    polarity = -1

                # Calculate indices
                origin_1_index = self.variables.index(origin_1)
                origin_2_index = self.variables.index(origin_2)
                destination_index = self.variables.index(destination)

                # Add polarity to interactions matrix
                self.interactions_matrix[destination_index, origin_2_index, origin_1_index] = polarity
        

    def adjacency_matrix_from_kumu(self):
        """Run the CLD analysis by extracting the adjacency matrix and interactions matrix."""
        self.extract_adjacency_matrix()
        self.extract_interactions_matrix()
        return self.variables, self.var_to_type, self.adjacency_matrix, self.interactions_matrix


### TESTING ###
    def test_extraction(self):
        """Test the CLD extraction by creating an examplar Kumu table and comparing the results."""
        # Create a sample evidence table
        data = {
            "From": ["A", "B", "C"],
            "Type": ["+", "-", "+"],
            "To": ["B", "C", "A"]
        }

        data_int = {
            "From1": ["A", "B"],
            "From2": ["C", "C"],
            "Type": ["+", "+"],
            "To": ["B", "A"]
        }

        df_e = pd.DataFrame(data["From"], columns=["Label"])
        df_e["Type"] = ["stock", "auxiliary", "constant"]
        df_c = pd.DataFrame(data)
        df_i = pd.DataFrame(data_int)

        # Save the evidence table to an Excel file
        original_file_path = self.file_path
        test_file_path = os.path.join(os.path.dirname(__file__), '..', 'test_files', 'evidence_table.xlsx')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

        with pd.ExcelWriter(test_file_path) as writer:
            df_e.to_excel(writer, sheet_name='Elements', index=False)
            df_c.to_excel(writer, sheet_name='Connections', index=False)
            df_i.to_excel(writer, sheet_name='Interactions', index=False)

        # Run the extraction
        self.file_path = test_file_path
        self.adjacency_matrix_from_kumu()
        self.file_path = original_file_path  # Set the original file path again
    
        # Define the expected results
        expected_adjacency_matrix = np.array([[0, 0, 1],
                                              [1, 0, 0],
                                              [0, -1, 0]])

        expected_interactions_matrix = np.array([[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 1, 0]], 
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [1, 0, 0]], 
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]]])

        # Assess the results
        assert np.all(expected_adjacency_matrix == self.adjacency_matrix)
        assert np.all(expected_interactions_matrix == self.interactions_matrix)
        assert np.all([x in self.variables for x in data["From"]])
        assert np.all([x in data["From"] for x in self.variables])
        print("Test for loading KUMU table passed.")
    