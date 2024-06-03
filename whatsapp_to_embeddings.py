import json
import sys
import csv
from datetime import datetime

WHATSAPP_WORKING_PATH = "<<<<path_to_whatsapp_working_folder (e.g. /Users/unixuser/whatsapp_working_folder)>>>"
CONTENT_LEN_PER_ITEM_THRESHOLD = 500

def parse(should_print):
    output_content_list = []

    contacts = get_contacts_from_csv()

    f = open(whatsapp_working_path + '/result.json')
    data = json.load(f)

    for key_conversation, conversation in data.items():
        phone_number = key_conversation.split("@", 1)[0]
        conversation_with_name = get_name_of_other_interlocutor_in_conversation(contacts, phone_number, conversation["name"])
        accumulator = []
        for key_message, message in conversation["messages"].items():
            is_from_me = message["from_me"]
            message_text = message["data"]
            datetime_object = datetime.fromtimestamp(message["timestamp"])
            formatted_date = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
            if (message_text is None or message_text.strip() == ""):
                continue
            formatted_message = f'message from me to "{conversation_with_name}", on {formatted_date}: {message_text}' if is_from_me else f'message from "{conversation_with_name}" to me, on {formatted_date}: {message_text}'
            accumulator.append(formatted_message)
            accumulated_messages = "\n".join(accumulator)
            if (should_print):
                print(formatted_message)
            if (len(accumulated_messages) > CONTENT_LEN_PER_ITEM_THRESHOLD):
                output_content_list.append((accumulated_messages, "<whatsapp>:" + key_conversation))
                accumulator = []
    f.close()

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
