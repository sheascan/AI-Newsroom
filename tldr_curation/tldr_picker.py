import os
import email
import email.policy
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, send_file
import datetime

# app = Flask(__name__)
# Force Flask to look in the correct folder, no matter where you run the script from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static") # Good practice to add this too

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
# We output directly to the MAIN studio input folder for seamless integration
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "inputs") 

def extract_links_from_eml(filepath):
    links = []
    try:
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)
        
        html_content = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try: html_content += part.get_content()
                    except: pass
        else:
            if msg.get_content_type() == "text/html":
                html_content = msg.get_content()

        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            for a in soup.find_all('a', href=True):
                text = a.get_text(strip=True)
                href = a['href']
                
                # Basic Noise Filter (Skip Unsubscribe, short labels, etc)
                if len(text) > 5 and "unsubscribe" not in text.lower() and "http" in href:
                    # Clean Tracking junk from URL (optional but nice)
                    if "utm_" in href:
                        href = href.split("?")[0]
                    
                    links.append({"text": text, "url": href})
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    return links

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_urls = request.form.getlist('selected_links')
        if selected_urls:
            filename = f"TLDR_Selection_{datetime.datetime.now().strftime('%d%b_%H%M')}.txt"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            with open(output_path, "w") as f:
                for url in selected_urls:
                    f.write(f"{url}\n")
            
            return f"""
            <div style='font-family:sans-serif; text-align:center; padding:50px;'>
                <h1 style='color:green;'>✅ Saved!</h1>
                <p>Saved <strong>{len(selected_urls)}</strong> articles to:</p>
                <p><code>{output_path}</code></p>
                <br>
                <a href='/'>Go Back</a>
            </div>
            """

    # --- GET REQUEST (Show the form) ---
    all_files_data = []
    
    if os.path.exists(INPUT_DIR):
        for f in os.listdir(INPUT_DIR):
            if f.endswith(".eml"):
                path = os.path.join(INPUT_DIR, f)
                extracted = extract_links_from_eml(path)
                if extracted:
                    all_files_data.append({"filename": f, "links": extracted})
    
    return render_template('index.html', files=all_files_data)

if __name__ == '__main__':
    print("🚀 TLDR Picker running at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)