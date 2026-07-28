import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv(override=True)
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)

account = client.api.accounts(account_sid).fetch()
print("account name:  ", account.friendly_name)
print("account type:  ", account.type)
print("account status:", account.status)
print()

owned_numbers = client.incoming_phone_numbers.list()
print("numbers owned: ", len(owned_numbers))
for number_record in owned_numbers:
    print("   ", number_record.phone_number, number_record.voice_url)
print()

visible_accounts = client.api.accounts.list()
print("accounts visible from these credentials:", len(visible_accounts))
for account_record in visible_accounts:
    print("   ", account_record.sid[:8] + "...", account_record.friendly_name, account_record.type)
