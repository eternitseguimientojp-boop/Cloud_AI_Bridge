import os
import base64
import json
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.cloud import firestore

app = Flask(__name__)

# Initialize Firestore DB globally
try:
    db = firestore.Client()
except Exception as e:
    print(f"WARNING: Failed to initialize Firestore Client. Error: {e}")
    db = None

# Basic API Key authentication
EXPECTED_API_KEY = os.environ.get('API_KEY')
if not EXPECTED_API_KEY:
    print("CRITICAL: API_KEY environment variable is not set. API authentication will fail.")

# Initialize the Gemini client using the google-genai SDK
api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    print("CRITICAL: GEMINI_API_KEY environment variable is not set.")
    client = None
else:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("WARNING: Failed to initialize Gemini Client. Error:", e)
        client = None

def require_api_key(func):
    def wrapper(*args, **kwargs):
        # In Cloud Functions, 'request' is a global flask.request
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        
        token = auth_header.split(' ')[1]
        print(f"AUTH Token: {token}")
        print(f"EXPECTED Token: {EXPECTED_API_KEY}")
        if token != EXPECTED_API_KEY:
            return jsonify({"error": "Unauthorized API Key"}), 401
            
        return func(*args, **kwargs)
    
    # Required to preserve the original function name for Flask routes
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/extract_pdf', methods=['POST'])
@require_api_key
def extract_pdf(arg_request=None):
    """
    HTTP Cloud Function.
    Args:
        arg_request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    try:
        # Use the passed argument if provided (Cloud Functions), 
        # otherwise fallback to the global flask request (Local Testing).
        actual_request = arg_request if arg_request is not None else request
        
        data = actual_request.get_json(silent=True)
        if not data or 'pdf_base64' not in data:
            return jsonify({"error": "Missing 'pdf_base64' in JSON body"}), 400

        pdf_base64 = data['pdf_base64']
        pdf_bytes = base64.b64decode(pdf_base64)
        customer_id = data.get('customer_id')
        new_instructions = data.get('new_instructions')
        
        # System prompt instructions
        prompt = """
        You are an AI assistant that extracts structured data from customer order PDFs. 
        Please locate and extract the following fields precisely:
        - purchase_order (string): The PO number.
        - full_delivery_address,  The full delivery address, usually the block below Ship To
        - delivery_address (string): The full delivery address (excluding zip code).
        - delivery_address_name, first line of delivery address block, excludin label Name
        - zip_code (string): The zip code of the delivery address.
        - materials (array of objects): Each item ordered, including:
            - item_number (string): The customer ID for the item.
            - description (string): The name or description of the item.
            - quantity (number): The quantity ordered.
            - unit_of_measure (string): The unit of measure (e.g., EA, Box, KG).
        
        Also provide:
        - confidence_score (number): A float between 0.0 and 1.0 indicating your confidence.
        - reasoning (string): A short explanation of any missing fields or difficulties.
        """

        # Look up or update customer-specific instructions globally
        applied_instructions = None
        if customer_id and db:
            doc_ref = db.collection('customer_instructions').document(customer_id)
            
            # If new instructions are provided, update the DB
            if new_instructions:
                try:
                    # Parse the incoming stringified JSON dictionary
                    new_instr_dict = json.loads(new_instructions)
                    
                    # Use set with merge=True to update existing fields or add new ones
                    doc_ref.set(new_instr_dict, merge=True)
                    print(f"Saved/Updated specific field instructions in Firestore for customer: {customer_id}")
                except Exception as e:
                    print(f"Warning: Could not save instructions to Firestore: {e}")

            # Fetch custom instructions
            try:
                doc = doc_ref.get()
                if doc.exists:
                    custom_instruction = doc.to_dict()
                    if custom_instruction and isinstance(custom_instruction, dict):
                        applied_instructions = custom_instruction
                        print(f"Applying custom field instructions for customer: {customer_id}")
                        prompt += "\n\nCRITICAL CUSTOMER-SPECIFIC INSTRUCTIONS:\n"
                        for field, instr in custom_instruction.items():
                            prompt += f"- {instr}\n"
            except Exception as e:
                print(f"Warning: Could not read instructions from Firestore: {e}")
        elif customer_id and not db:
            print(f"WARNING: Cannot apply instructions for {customer_id} because Firestore is not initialized.")

        # Set up response schema to enforce JSON output structure
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "purchase_order": {"type": "STRING", "nullable": True},
                "full_delivery_address": {"type": "STRING", "nullable": True},
                "delivery_address": {"type": "STRING", "nullable": True},
                "delivery_address_name": {"type": "STRING", "nullable": True},
                "zip_code": {"type": "STRING", "nullable": True},
                "materials": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "item_number": {"type": "STRING", "nullable": True},
                            "description": {"type": "STRING", "nullable": True},
                            "quantity": {"type": "NUMBER", "nullable": True},
                            "unit_of_measure": {"type": "STRING", "nullable": True}
                        }
                    },
                    "nullable": True
                },
                "confidence_score": {"type": "NUMBER"},
                "reasoning": {"type": "STRING", "nullable": True}
            },
            "required": ["confidence_score"]
        }

        # Generate content using Gemini 2.5 Flash
        try:
            print("Sending request to base Gemini model...")
            
            # Re-attempt initialization if it failed globally
            global client
            if client is None:
                print("Attempting to re-initialize Gemini client...")
                client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
                
            BASE_MODEL = 'gemini-2.5-flash'
            PRO_MODEL = 'gemini-2.5-pro'
            CONFIDENCE_THRESHOLD = 0.8
            
            response = client.models.generate_content(
                model=BASE_MODEL, 
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.1, # Low temperature for extraction tasks
                )
            )
            print("Successfully received response from base Gemini")
            
            extracted_json = json.loads(response.text)
            
            # Check confidence for fallback
            confidence = extracted_json.get('confidence_score', 0)
            if confidence < CONFIDENCE_THRESHOLD:
                print(f"Confidence score {confidence} is below threshold {CONFIDENCE_THRESHOLD}. Falling back to Pro model...")
                
                correction_prompt = f"""
                The previous base model extracted this data from the PDF but was unsure and had a low confidence score of {confidence}. 
                Can you carefully review the original PDF and correct any errors in this extraction?
                
                Previous Extracted Data:
                {json.dumps(extracted_json, indent=2)}
                
                Original Extraction Instructions to adhere to:
                {prompt}
                """
                
                pro_response = client.models.generate_content(
                    model=PRO_MODEL, 
                    contents=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                        correction_prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1, 
                    )
                )
                print("Successfully received response from Pro Gemini")
                extracted_json = json.loads(pro_response.text)
                
        except Exception as api_err:
            import traceback
            print("CRITICAL: Gemini model error (API not enabled, invalid key, or transient failure):")
            traceback.print_exc()
            # Generic error to caller, detailed log for IT
            return jsonify({"error": "An internal error occurred while processing the AI extraction. Please contact IT support."}), 500

        if applied_instructions:
            extracted_json['applied_instructions'] = applied_instructions

        return jsonify(extracted_json), 200

    except Exception as e:
        import traceback
        print("CRITICAL: General extraction route failure:")
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error. Please contact IT support."}), 500

if __name__ == '__main__':
    # Run the Flask app on port 8080 (for local testing)
    app.run(host='0.0.0.0', port=8080)
