
import streamlit as st
import boto3
import json
import pefile
import os
import tempfile

from scipy.sparse import hstack, csr_matrix
import collections
from nltk import ngrams
import numpy as np

N = 2

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary directory and return the file path."""
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def byteSequenceToNgrams(byteSequence, n):
    """Convert byte sequence to n-grams."""
    Ngrams = ngrams(byteSequence, n)
    return list(Ngrams)

def extractNgramCounts(file_data, N):
    """Extract n-gram counts from the file."""
    fileNgrams = byteSequenceToNgrams(file_data, N)
    return collections.Counter(fileNgrams)

def getNGramFeaturesFromSample(file_data, K1_most_common_Ngrams_list):
    """Get n-gram features from the file."""
    K1 = len(K1_most_common_Ngrams_list)
    fv = K1 * [0]
    fileNgrams = extractNgramCounts(file_data, N)
    for i in range(K1):
        fv[i] = fileNgrams[K1_most_common_Ngrams_list[i]]
    return fv

def preprocessImports(listOfDLLs):
    """Preprocess the list of DLLs."""
    processedListOfDLLs = []
    temp = [x.decode().split(".")[0].lower() for x in listOfDLLs]
    return " ".join(temp)

def getImports(pe):
    """Get imports from the PE file."""
    listOfImports = []
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        listOfImports.append(entry.dll)
    return preprocessImports(listOfImports)

def getSectionNames(pe):
    """Get section names from the PE file."""
    listOfSectionNames = []
    for eachSection in pe.sections:
        refined_name = eachSection.Name.decode().replace('\x00','').lower()
        listOfSectionNames.append(refined_name)
    return " ".join(listOfSectionNames)

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime',
                      aws_access_key_id='ASIAU6GD32CRPCW7UHNX',
                      aws_secret_access_key='Rzr5wWPJPDllWJpL92TaWatwwUB2ZeOWYHBaM+kk',
                      aws_session_token='IQoJb3JpZ2luX2VjEFsaCXVzLXdlc3QtMiJGMEQCIDoOmBvyP6pTVsaLLC0wjE1d/DWuSfTwvbXRXllzUXIeAiAG4aX9fcWVd7IMBT/h0dkR9jV03Xy5aDf9QaYS08elGCq7AgiE//////////8BEAAaDDMzOTcxMzE4MzkwNiIMFpZLS0/rY9yhCHdMKo8C7LcF2wfCmx+3FF95K4GgqfdVBjPctE2GYY4m3LDIOnSFNKFESgcErt7hS4+scXhAYRKyVwqsQA/kOJ+9N9R/FjktMsmQGT2hBd/TN8ovk1X3XB0BcmtIRYy+ZTN4OnQAytU2g2UV1ybmdj+GkhgTt2ncC4tH7Fp1aDf6nm1EA4qiTPyOzM2V+C+bMjbTqmpyq2l/FRf8rj3VifSzN5llshbyQdUYrMRDtG3syEOAqocVlqGUrokHV6iL01aeeCzkyzzDl+uixtSO9R92nsBXMJBLl0U3z6bpbZGnVJ+HsqPuWTm83IDo6g+uYN60wfujy7hId63NWam6FM7Hmf4wB7F3jEY/rHWrB/PDgb3PCTDep7iwBjqeAebKYOSVu7wMcTTruOnS4hEgxV46k8Y0uSqkWCZzFC1Ipo2lb0xrL2ALH3wt0awM8iLxULr4WvKmLUO5WQpbdOQza3l2aMBzZ1CdgegpryaHlTd9lSshwmsVUAy5t/bItN2WBbyrSK/2GeRlgYtOSzf3a4Eh2hcJ9yiFB+3Zp75kQLOinm/N3UtCAfvtEQngx1lCg7p2WS9m5EvrMlpx==',
                      region_name='us-east-1')

# Function to extract features from the .exe file
def extract_features(uploaded_file):
    # Extract features using pefile, getImports, getSectionNames
    importsCorpus_pred = []
    numSections_pred = []
    sectionNames_pred = []
    NgramFeaturesList_pred = []
    K1_most_common_Ngrams_list = [(0, 0),
 (255, 255),
 (204, 204),
 (2, 100),
 (1, 0),
 (0, 139),
 (131, 196),
 (2, 0),
 (68, 36),
 (139, 69),
 (0, 131),
 (255, 117),
 (133, 192),
 (255, 139),
 (254, 255),
 (46, 46),
 (139, 77),
 (141, 77),
 (255, 21),
 (7, 0),
 (69, 252),
 (8, 139),
 (76, 36),
 (0, 1),
 (4, 0),
 (4, 139),
 (137, 69),
 (141, 69),
 (0, 137),
 (0, 255),
 (255, 131),
 (51, 192),
 (80, 232),
 (255, 141),
 (85, 139),
 (8, 0),
 (3, 100),
 (0, 232),
 (15, 182),
 (0, 116),
 (139, 236),
 (64, 0),
 (80, 141),
 (15, 132),
 (12, 139),
 (100, 0),
 (253, 255),
 (255, 0),
 (84, 36),
 (73, 78),
 (65, 68),
 (0, 204),
 (80, 65),
 (68, 68),
 (78, 71),
 (68, 73),
 (16, 0),
 (198, 69),
 (192, 116),
 (199, 69),
 (80, 255),
 (204, 139),
 (2, 101),
 (4, 137),
 (139, 68),
 (116, 36),
 (3, 0),
 (0, 8),
 (139, 76),
 (106, 0),
 (101, 0),
 (196, 12),
 (100, 139),
 (139, 70),
 (64, 2),
 (36, 8),
 (0, 89),
 (69, 8),
 (117, 8),
 (196, 4),
 (86, 139),
 (95, 94),
 (139, 255),
 (32, 0),
 (0, 16),
 (131, 192),
 (0, 80),
 (0, 141),
 (195, 204),
 (36, 20),
 (36, 16),
 (0, 117),
 (139, 240),
 (9, 0),
 (100, 232),
 (0, 128),
 (6, 0),
 (8, 137),
 (1, 100),
 (131, 248)]

    # Save the uploaded file to the local filesystem
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        # Extract features from the temporary file
        NGramFeatures_pred = getNGramFeaturesFromSample(temp_file_path, K1_most_common_Ngrams_list)
        pe_pred = pefile.PE(temp_file_path)
        imports_pred = getImports(pe_pred)
        nSections_pred = len(pe_pred.sections)
        secNames_pred = getSectionNames(pe_pred)
        importsCorpus_pred.append(imports_pred)
        numSections_pred.append(nSections_pred)
        sectionNames_pred.append(secNames_pred)
        NgramFeaturesList_pred.append(NGramFeatures_pred)
        importsCorpus_pred = " ".join(importsCorpus_pred)
        sectionNames_pred = " ".join(sectionNames_pred)
        numSections_pred = numSections_pred[0]
        print(NgramFeaturesList_pred, importsCorpus_pred, sectionNames_pred, numSections_pred)
        return {
            "NgramFeaturesList_pred": NgramFeaturesList_pred,
            "importsCorpus_pred": importsCorpus_pred,
            "sectionNames_pred": sectionNames_pred,
            "numSections_pred": str(numSections_pred)
        }
    finally:
        # Clean up: delete the temporary file
        os.remove(temp_file_path)

# Function to send features to SageMaker endpoint for inference
def invoke_endpoint(features):
    # Serialize features to JSON
    payload = json.dumps(features)

    # Specify your endpoint name
    endpoint_name = "sklearn-local-ep2024-04-03-20-45-08"
    # Send inference request to the endpoint
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType='application/json',
                                       Body=payload)

    # Parse the prediction response
    result = json.loads(response['Body'].read().decode())

    return result

# Streamlit app
def main():
    st.title("SageMaker Inference with Streamlit")

    # File upload widget
    uploaded_file = st.file_uploader("Upload .exe file", type="exe")

    if uploaded_file is not None:
        # Perform feature extraction
        features = extract_features(uploaded_file)

        # Perform inference
        prediction = invoke_endpoint(features)

        # Display prediction
        #st.write("Prediction:", prediction)
        if prediction["Output"] == 0:
            st.success("Benign - Safe")
        else:
            st.error("Malware - Danger")

if __name__ == "__main__":
    main()


