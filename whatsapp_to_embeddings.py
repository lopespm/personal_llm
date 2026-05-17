import json
import sys
import csv
from datetime import datetime
from tqdm import tqdm

WHATSAPP_WORKING_PATH = "<<<<path_to_whatsapp_working_folder (e.g. /Users/unixuser/whatsapp_working_folder)>>>"
CONTENT_LEN_PER_ITEM_THRESHOLD = 500

def parse(should_print):
    output_content_list = []

    contacts = get_contacts_from_csv()

    with open(WHATSAPP_WORKING_PATH + '/result.json') as f:
        data = json.load(f)

    for key_conversation, conversation in tqdm(data.items(), desc='Parsing WhatsApp', unit='chat'):
        phone_number = key_conversation.split("@", 1)[0]
        conversation_with_name = get_name_of_other_interlocutor_in_conversation(contacts, phone_number, conversation["name"])
        accumulator = []
        accumulated_len = 0
        for key_message, message in conversation["messages"].items():
            is_from_me = message["from_me"]
            message_text = message["data"]
            datetime_object = datetime.fromtimestamp(message["timestamp"])
            formatted_date = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
            if message_text is None or message_text.strip() == "":
                continue
            formatted_message = f'message from me to "{conversation_with_name}", on {formatted_date}: {message_text}' if is_from_me else f'message from "{conversation_with_name}" to me, on {formatted_date}: {message_text}'
            accumulator.append(formatted_message)
            accumulated_len += len(formatted_message) + 1  # +1 for joining \n
            if should_print:
                print(formatted_message)
            if accumulated_len > CONTENT_LEN_PER_ITEM_THRESHOLD:
                output_content_list.append(("\n".join(accumulator), "<whatsapp>:" + key_conversation))
                accumulator = []
                accumulated_len = 0

    return output_content_list

def get_contacts_from_csv():
    contacts = dict()
    with open(WHATSAPP_WORKING_PATH + '/contacts.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            phone1 = row['Phone 1 - Value']
            phone2 = row['Phone 2 - Value']
            if (phone1 is not None):
                contacts[phone1.replace(' ', '').replace('+', '').replace('-', '')] = row['Name']
            if (phone2 is not None):
                contacts[phone2.replace(' ', '').replace('+', '').replace('-', '')] = row['Name']
    return contacts

def get_name_of_other_interlocutor_in_conversation(contacts, phone_number, exported_conversation_name) -> str:
    if (exported_conversation_name is not None):
        return exported_conversation_name
    elif (phone_number in contacts):
        return contacts[phone_number]
    elif (phone_number[:3] == "351" and  phone_number[3:] in contacts):
        return contacts[phone_number[3:]]
    else:
        return ""

def main():
    parse(True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
