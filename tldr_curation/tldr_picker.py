import os
import email
import email.policy
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
import datetime # Standard datetime import

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "inputs") 

# TIME LIMIT (Hours)
AGE_LIMIT_HOURS = 36

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

def extract_links_from_eml(filepath):
    links = []
    email_display_date = "Unknown Date" 
    
    try:
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)
        
        # --- 1. Extract Date & Check Age ---
        if msg['Date']:
            try:
                dt = parsedate_to_datetime(msg['Date'])
                
                # Get current time matching the email's timezone info (if available)
                # This ensures we compare "apples to apples"
                now = datetime.datetime.now(dt.tzinfo) if dt.tzinfo else datetime.datetime.now()
                
                # AGE CHECK: If older than 36 hours, skip immediately
                if (now - dt) > datetime.timedelta(hours=AGE_LIMIT_HOURS):
                    print(f"🚫 Ignoring {os.path.basename(filepath)} (Older than {AGE_LIMIT_HOURS}h)")
                    return [], None

                # Format for display: "22 Dec 08:30"
                email_display_date = dt.strftime('%d %b %H:%M') 
            except Exception as e:
                print(f"Date parse error in {filepath}: {e}")

        # --- 2. Extract Links ---
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
                
                # Basic Noise Filter
                if len(text) > 5 and "unsubscribe" not in text.lower() and "http" in href:
                    if "utm_" in href:
                        href = href.split("?")[0]
                    
                    if " minute read)" in text.lower():
                        links.append({"text": text, "url": href})
                        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return links, email_display_date

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_urls = request.form.getlist('selected_links')
        if selected_urls:
            timestamp = datetime.datetime.now().strftime('%d%b_%H%M')
            filename = f"TLDR_Selection_{timestamp}.txt"
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

    # --- GET REQUEST ---
    all_files_data = []
    
    if os.path.exists(INPUT_DIR):
        for f in os.listdir(INPUT_DIR):
            if f.endswith(".eml"):
                path = os.path.join(INPUT_DIR, f)
                extracted_links, email_date = extract_links_from_eml(path)
                
                # Only add to list if links exist (age check returns [] if too old)
                if extracted_links:
                    all_files_data.append({
                        "filename": f,
                        "date": email_date,
                        "links": extracted_links
                    })
    
    return render_template('index.html', files=all_files_data)

if __name__ == '__main__':
    print("🚀 TLDR Picker running at: http://127.0.0.1:5000")
    print(f"🕒 Ignoring emails older than {AGE_LIMIT_HOURS} hours.")
    app.run(debug=True, port=5000)