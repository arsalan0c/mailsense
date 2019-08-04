#!/usr/bin/env python3

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient import errors

from datetime import datetime, timezone
import os.path
import pickle
import ast
import operator
import time
import logging

from sentimentanalysis import Model
from metrics import metrics

def get_service(credentials_path, token_path):
	''' Constructs a Gmail service object to use the Gmail API.
		Makes use of a credentials and token file (separate).

		Args:
			credentials_path: Path to a file with Google API credentials.
			token_path: Path to a pickle file with  user's access and refresh tokens.

		Returns:
			service: Gmail service object.
	'''
	scopes = ['https://www.googleapis.com/auth/gmail.modify']

	creds = None
	# The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
	if os.path.exists(token_path):
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

def get_mail_id(service, history_id):
	'''Returns the id of a new Gmail mail based on a message received by a Gmail subscriber.

	Retrieves the history of messages added to the Gmail inbox.
	Returns the id of the first one in the history.

	Args:
		service: A Gmail service object to access the Gmail API.
		history_id: History id of a message received by a Gmail subscriber parsed into a dictionary.
	'''
	try:
		history_obj = service.users().history().list(userId='me', historyTypes='messageAdded', startHistoryId=history_id).execute()
		mail_id = history_obj['history'][0]['messages'][0]['id']
		return mail_id
	except (KeyError, errors.HttpError) as e:
		# can be due to no new messages
		logger.warning('no mail id found for message with history id: %s', history_id, exc_info=True)

def get_mail_texts(service, mail_id, history_id):
	'''Returns selected texts of a Gmail mail using its unique identifier.

	Retrieves the subject and a snippet of the mail's body.
	Returns them in a list if they can be retrieved.

	Args:
		service: A Gmail service object to access the Gmail API.
		mail_id: Unique identifier of a mail to retrieve texts for.
		history_id: History id of a message received by a Gmail subscriber parsed into a dictionary.
	'''
	try:
		mail_obj = service.users().messages().get(userId='me', id=mail_id).execute()
		mail_subject = mail_obj['payload']['headers'][3]['value']
		mail_snippet = mail_obj['snippet']
		return [mail_subject, mail_snippet]
	except (KeyError, errors.HttpError) as e:
		logger.error('no texts found for mail id %s for message with history id: ', mail_id, history_id, exc_info=True)

def inference(mail_texts):
	'''Returns the polarity label of a mail based on its texts.

	Performs inference on each of the mail's texts.
	Returns the polarity label with the highest count as the mail's polarity label.

	Args:
		mail_texts: A List of texts of a mail to perform inference on. For example, subject and body.
	'''
	polarity_counts = {}
	for text in mail_texts:
		if len(text) > 0:
			polarity_label = model.predict(text)
			# increment count of this polarity type
			polarity_counts[polarity_label] = 1 + polarity_counts.get(polarity_label, 0)

	return max(polarity_counts.items(), key=operator.itemgetter(1))[0]

def get_label_id(service, label_name):
	'''Returns the unique identifier for a Gmail label based on its name.

	Retrieves all the labels in the authenticated user's Gmail.
	Returns the id of the first label whose name matches.
	Creates the label if there is no match and returns its id.

	Args:
		service: A Gmail service object to access the Gmail API.
		label_name: Name of the label whose id is to be retrieved.
	'''
	# all present labels
	labels = service.users().labels().list(userId='me').execute().get('labels', [])
	for label in labels:
		if label['name'] == label_name:
			return label['id']

	created_label = service.users().labels().create(userId='me', body={'name': label_name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}).execute()
	return created_label['id']

def assign_label(service, mail_id, polarity_label):
	'''Assigns a label to a Gmail mail.

	Determines the label to be assigned, as a combination of the polarity text and its corresponding emoji.
	Assigns the label to the relevant mail using the mail's id and the label's id.

	Args:
		service: A Gmail service object to access the Gmail API.
		mail_id: Unique identifier for a Gmail mail to be assigned a label to.
		polarity_label: Polarity label of the mail, a result of the classification of its text(s).
	'''
	label_id = get_label_id(service, polarity_label)
	# assign label to mail
	labels_to_change = {'removeLabelIds': [], 'addLabelIds': [label_id]}
	service.users().messages().modify(userId='me', id=mail_id, body=labels_to_change).execute()

def process_message(service, message):
	'''Orchestrator, to processes a message received by a Gmail subscriber.

	Calls functions to achieve the following (in order):
		Retrieve the id of a new mail in an inbox, if any.
		Retrieve relevant texts of the new mail using the id.
		Retrieve inference on the texts.
		Assign a label to the new mail.

	Args:
		service: A Gmail service object to access the Gmail API.
		message: A message received by a Gmail subscriber.
	'''
	message_dict = ast.literal_eval(message.data.decode('utf-8'))
	history_id = message_dict['historyId']
	logger.info('received message with history id: %s', history_id)

	# buffer to allow Gmail API provide up-to-date results
	time.sleep(1)

	mail_id = get_mail_id(service, history_id)
	if mail_id is None:
		return
	logger.info('retrieved mail id %s for message with history id: %s', mail_id, history_id)

	mail_texts = get_mail_texts(service, mail_id, history_id)
	if mail_texts is None:
		return
	logger.info('retrieved texts %s for mail from message with history id: %s', mail_texts, history_id)

	polarity_label = inference(mail_texts)
	logger.info('retrieved polarity label %s for mail from message with history id: %s', polarity_label, history_id)

	assign_label(service, mail_id, polarity_label)
	logger.info('assigned label %s to mail from message with history_id: %s\n', polarity_label, history_id)
	try:
		mail_stats.addPolarity(polarity_label)
	except Error as e:
		logger.error('failed to record email polarity classification', exc_info=True)

def start(model_type, model_args, log_path):
	'''Performs initialization tasks.

	Initializes logger.
	Triggers the initialization of the polarity classification model.
	Defines global variables to be used.

	Args:
		model_type: A ModelType (sentimentanalysis.py) value of the sentiment analysis model to use.
		model_args: The argument for the respective sentiment analysis model.
		log_path: The path of the log file for this file.
	'''
	logging.basicConfig(filename=log_path, level=logging.INFO, format=str(datetime.now(timezone.utc).astimezone()) + ' %(name)s' + ' %(levelname)s-%(message)s')
	global logger
	logger = logging.getLogger('mailsense.mail.mail')
	logger.info('initializing')

	global mail_stats
	try:
		mail_stats = metrics()
	except Error as e:
		logger.error('failed to initialize mail statistics', exc_info=True)
		raise

	global model
	try:
		model = Model(model_type, model_args)
	except Error as e:
		logger.error('failed to initialize sentiment analysis model', exc_info=True)
		raise

	logger.info('initialization complete')
