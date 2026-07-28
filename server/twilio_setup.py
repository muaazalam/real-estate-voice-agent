import os, sys
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv(override=True)
client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

owned_numbers = client.incoming_phone_numbers.list()
print("numbers owned:", len(owned_numbers))
for number_record in owned_numbers:
    print("   ", number_record.phone_number, number_record.voice_url or "(no voice url)")

verified_caller_ids = client.outgoing_caller_ids.list()
print("verified caller IDs:", len(verified_caller_ids))
for caller_id_record in verified_caller_ids:
    print("   ", caller_id_record.phone_number)
print()

if owned_numbers:
    print("You already have a number. Trial gets one. Nothing to buy.")
    sys.exit()

if not verified_caller_ids:
    print("STOP: no verified caller ID. Twilio will not sell you a number.")
    print("Console > Phone Numbers > Verified Caller IDs > add your mobile, then rerun.")
    sys.exit()

area_code = sys.argv[1] if len(sys.argv) > 1 else None
search_arguments = {"voice_enabled": True, "limit": 10}
if area_code:
    search_arguments["area_code"] = area_code

print("Available voice-enabled numbers:")
for available_number in client.available_phone_numbers("US").local.list(**search_arguments):
    print("   ", available_number.phone_number, available_number.locality)
