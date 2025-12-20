import imaplib
import email
import os
import datetime
from email.header import decode_header
import re
import sys

# --- CONFIGURATION ---
IMAP_SERVER = "imap.gmail.com"
EMAIL_USER = "mike.mcconigley@gmail.com"
EMAIL_PASS = "rizj dfcd hywm pflx"

# Where to save the raw .eml files for the Web Picker
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")

# Archive folder on Gmail (created if missing)
ARCHIVE_FOLDER = "A_TLDR_Curator"

# LIST OF SENDERS TO HARVEST
# Add as many as you like here. No complex syntax required.
TARGET_SENDERS = [
    "dan@tldrnewsletter.com",
    "dan@tldr.tech",
    "newsletters@theregister.com"
]

# ---------------------

def clean_filename(subject):
    if not subject: return "unknown_subject"
    clean = re.sub(r'[^\w\s-]', '', subject)
    clean = re.sub(r'[-\s]+', '_', clean).strip()
    return clean[:50]

def connect_imap():
    if not EMAIL_USER or not EMAIL_PASS:
        print("❌ Error: Environment variables EMAIL_USER or EMAIL_PASS not set.")
        sys.exit(1)
    
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    return mail

def ensure_archive_folder(mail, folder_name):
    status, folders = mail.list()
    folder_exists = False
    for f in folders:
        # Decode folder name if necessary, though list returns bytes
        if folder_name.encode() in f:
            folder_exists = True
            break
    
    if not folder_exists:
        print(f"   ℹ️ Creating archive folder: '{folder_name}'...")
        mail.create(folder_name)
    return folder_name

def main():
    print("📨 Scanning for Curator Newsletters...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mail = connect_imap()
    mail.select("inbox")
    
    ensure_archive_folder(mail, ARCHIVE_FOLDER)

    # --- ROBUST SEARCH LOOP ---
    # Instead of one complex 'OR' query, we search one by one.
    all_email_ids = set()
    
    for sender in TARGET_SENDERS:
        # Search for emails FROM this sender
        # We use the simple criterion: FROM "email@address.com"
        typ, msg_data = mail.search(None, f'(FROM "{sender}")')
        
        if msg_data and msg_data[0]:
            # Add found IDs to our unique set
            ids = msg_data[0].split()
            all_email_ids.update(ids)
            # print(f"   -> Found {len(ids)} from {sender}")

    if not all_email_ids:
        print("   ✅ No new newsletters found.")
        mail.logout()
        return

    print(f"   -> Processing {len(all_email_ids)} unique emails...")

    for e_id in all_email_ids:
        try:
            # Fetch the email content
            res, msg_data = mail.fetch(e_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Decode Subject
            subject_header = msg["Subject"]
            if subject_header:
                decoded_list = decode_header(subject_header)
                subject = ""
                for part, encoding in decoded_list:
                    if isinstance(part, bytes):
                        subject += part.decode(encoding if encoding else "utf-8", errors="ignore")
                    else:
                        subject += part
            else:
                subject = "No Subject"
            
            # Create Filename
            date_prefix = datetime.datetime.now().strftime("%d%b")
            safe_subject = clean_filename(subject)
            filename = f"{date_prefix}_{safe_subject}.eml"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Save to Curator Input folder
            with open(filepath, "wb") as f:
                f.write(raw_email)
            
            print(f"      ✅ Saved: {filename}")

            # Move to Archive
            mail.copy(e_id, ARCHIVE_FOLDER)
            mail.store(e_id, '+FLAGS', '\\Deleted')

        except Exception as e:
            print(f"      ⚠️ Error processing email ID {e_id}: {e}")

    # Expunge (Permanently remove from Inbox)
    mail.expunge()
    mail.close()
    mail.logout()
    print(f"   📦 Archived all processed emails to '{ARCHIVE_FOLDER}'")

if __name__ == "__main__":
    main()