

import os 
from twilio.rest import Client
import dotenv
dotenv.load_dotenv()



#def text_message():
client = Client(os.getenv("Account_Sid"),
os.getenv("Auto_Token"))

"""
message = client.messages.create(
to = value
#os.getenv("Cell_phone"),
from_ = os.getenv("Phone_number"),
body = f"El electrodomestico identificado es safrdgr vf")

print(message.sid)    

"""

#num=["+34606515482","+34685204379", "+34690210228"]

#for i in range(0,len(num)):
 #   message = client.messages.create(to = num[i], from_=os.getenv("Phone_number"),
                       