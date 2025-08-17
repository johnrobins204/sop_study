import os
import PyPDF2

def extract_text_from_pdfs(pdf_dir="", output_dir=""):
    """
    Extracts full text content from all PDF files in a given directory
    and saves them as .txt files in an output directory
    This script is designed to solve the problem of missing tables and other
    content when manually copy-pasting from PDFs.

    Args:
        pdf_dir (str): The directory containing the source PDF files.
        output_dir (str): The directory where the extracted .txt files will be saved.
    """
    print("--- Starting PDF Text Extraction ---")

    # 2. Get the list of PDF files to process
    pdf_files = [f for f in os.listdir(pdf_dir or '.') if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ WARNING: No PDF files found in the '{pdf_dir}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")
    # 3. Loop through each PDF, extract text, and save it
    for pdf_filename in pdf_files:
        try:
            pdf_path = os.path.join(pdf_dir, pdf_filename)
            
            # Construct the output filename to match the main script's expectations
            output_filename = f"{pdf_filename}.txt"
            output_path = os.path.join(output_dir, output_filename)

            print(f"\nProcessing '{pdf_filename}'...")

            full_text = ""
            with open(pdf_path, 'rb') as pdf_file_obj:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                
                # Loop through all the pages and extract text
                for page_num in range(len(pdf_reader.pages)):
                    page_obj = pdf_reader.pages[page_num]
                    full_text += page_obj.extract_text()
            
            # 4. Save the extracted text to the output file
            with open(output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(full_text)
            
            print(f"✅ Successfully extracted text to '{output_path}'")

        except Exception as e:
            print(f"❌ ERROR processing '{pdf_filename}': {e}")

    print("\n--- Extraction Complete ---")

if __name__ == "__main__":
    extract_text_from_pdfs()
