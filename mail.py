#!/usr/bin/env python3

import os.path
import pickle

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def get_service(credentials_path, token_path):
	''' Constructs a Gmail service object to use the Gmail API.
		Makes use of a credentials as well as token file.

		Args:
			credentials_path: Path to a file with Google API credentials.
			token_path: Path to a pickle file with  user's access and refresh tokens.

		Returns:
			service: Gmail service object.
	'''
	scopes = ['https://mail.google.com/']

	creds = None
	if os.path.exists(credentials_path):
		with open(token_path, 'rb') as token:
			creds = pickle.load(token)

	# If there are no (valid) credentials available, let the user log in.
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_secrets_file(
				credentials_path, scopes)
			creds = flow.run_local_server()
		# Save the credentials for the next run
		with open(token_path, 'wb') as token:
			pickle.dump(creds, token)

	service = build('gmail', 'v1', credentials=creds)
	return service





