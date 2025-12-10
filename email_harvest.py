import imaplib
import email
import os
import datetime
from email.header import decode_header

# --- CONFIGURATION ---
IMAP_SERVER = "imap.gmail.com" # or outlook.office365.com
EMAIL_USER = "mike.mcconigley@gmail.com"
EMAIL_PASS = "rizj dfcd hywm pflx" 

# WHERE TO SAVE
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inputs")

# TARGET LIST: Add the newsletters you want to grab
# Leave 'subject' as None to match only by sender
TARGETS = [
    {"sender": "info@editorial.theguardian.com", "subject": None},
    {"sender": "newsletters@theguardian.com", "subject": "First Edition"},
 #   {"sender": "info@editorial.theguardian.com", "subject": "First Edition"},
 #   {"sender": "info@editorial.theguardian.com", "subject": "The Guardian Headlines:"},
    {"sender": "nytdirect@nytimes.com", "subject": "The World:"}, 
    {"sender": "techpresso@dupple.com", "subject": None},
    {"sender": "irishtimesinsidepolitics@comms.irishtimes.com", "subject": None},
    {"sender": "irishtimesmorningbriefing@comms.irishtimes.com", "subject": None},
    {"sender": "irishtimessportsbriefing@comms.irishtimes.com","subject": None},
    {"sender": "onthemoneytheirishtimes@comms.irishtimes.com","subject": None},
    {"sender": "newsletter@news.metro.co.uk","subject":"Your daily football updates are here"},
    {"sender": "@news.theregister.co.uk","subject":None},
]

def clean_filename(subject):
    return "".join(c if c.isalnum() else "_" for c in subject)[:50]

def fetch_emails():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # Connect
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    today = datetime.datetime.now().strftime("%d-%b-%Y")
    print(f"📨 Scanning Inbox for {today}...")

    for target in TARGETS:
        criteria = []
        # IMAP Search is tricky, we build the query dynamically
        # Searching for emails 'SINCE' yesterday to ensure we catch morning papers
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%d-%b-%Y")
        
        query = f'(SINCE "{yesterday}" FROM "{target["sender"]}")'
        
        status, messages = mail.search(None, query)
        
        if status != "OK": continue
        
        email_ids = messages[0].split()
        print(f"   -> Found {len(email_ids)} emails from {target['sender']}")

        for e_id in email_ids:
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Decode Subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    
                    # Filter by subject if required
                    if target["subject"] and target["subject"].lower() not in subject.lower():
                        continue

                    # Save File
                    safe_sub = clean_filename(subject)
                    filename = f"{today}_{safe_sub}.eml"
                    filepath = os.path.join(SAVE_DIR, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(response_part[1])
                    print(f"      ✅ Saved: {filename}")

    mail.close()
    mail.logout()

if __name__ == "__main__":
    fetch_emails()
