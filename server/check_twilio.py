import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv(override=True)

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

print("SID:   ", (account_sid[:6] + "...") if account_sid else "MISSING FROM .env")
print("Token: ", "{} chars".format(len(auth_token)) if auth_token else "MISSING FROM .env")

client = Client(account_sid, auth_token)

for number_record in client.incoming_phone_numbers.list():
    print()
    print("number:      ", number_record.phone_number)
    print("sid:         ", number_record.sid)
    print("voice url:   ", number_record.voice_url or "(not set)")
    print("voice method:", number_record.voice_method or "(not set)")
