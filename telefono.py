

import os 
from twilio.rest import Client
import dotenv
dotenv.load_dotenv()
from audio import reconocedor_audio




def text_message():
    client = Client(os.getenv("Account_Sid"),
    os.getenv("Auto_Token"))


    message = client.messages.create(
    to = os.getenv("Cell_phone"),
    from_ = os.getenv("Phone_number"),
    body = f"El electrodomestico identificado es : {result[int(y_pref_sol[-1])]}")

    print(message.sid)

