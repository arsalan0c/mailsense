#!/usr/bin/env python3

# This script is used to run a subscriber for a Gmail inbox.
# It is also used to the enact the functionality provided by mail to process subscription messages.

from google.cloud import pubsub_v1

import time
import argparse
import ast
import os
import sys

import mail
from watch import watch
from sentimentanalysis import ModelType
from sentimentanalysis import MODEL_ARGUMENT_CHOICES

def callback(message):
	'''Receives a Gmail subscription message and processes it.

	Acknowledges the message.
	Creates a service object to access the Gmail API, to process the message.
	Uses mail to process the message.

	Args:
		message: A Gmail subscription message
	'''
	# respond with acknowledgement, otherwise the message will keep being received
	message.ack()

	# create a new service object every time because it is not thread safe
	service = mail.get_service(args.credentialspath, args.tokenpath)
	mail.process_message(service, message)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='process arguments required for subscriber and mail functionality')
	parser.add_argument('-p', '--project', help='string: name of the project from Google Cloud', type=str, action='store', required=True)
	parser.add_argument('-s', '--subscription', help='string: name of the subscription from Google Cloud', type=str, action='store', required=True)
	parser.add_argument('-t', '--topic', help='string: name of the topic from Google Cloud', type=str, action='store', required=True)
	parser.add_argument('-mt', '--modeltype', help='string: the type of model to initialize', type=lambda model: ModelType[model], choices=list(ModelType), action='store', required=True)
	parser.add_argument('-ma', '--modelargs', help='dict: the arguments to initialize the respective model. Arguments required for the various model types: ' + str(MODEL_ARGUMENT_CHOICES), type=ast.literal_eval, action='store', required=False)
	parser.add_argument('-cp', '--credentialspath', help='string: path to Gmail API credentials file', type=str, action='store', required=True)
	parser.add_argument('-tp', '--tokenpath', help='string: path to Gmail API token file', type=str, action='store', required=True)
	args = parser.parse_args()

	subscriber = pubsub_v1.SubscriberClient()
	# The `subscription_path` method creates a fully qualified identifier
	# in the form `projects/{project_id}/subscriptions/{subscription_name}`
	subscription_path = subscriber.subscription_path(args.project, args.subscription)

	service = mail.get_service(args.credentialspath, args.tokenpath)
	watch(service, args.project, args.topic)

	log_dir = os.path.join(os.path.dirname(__file__), '../logs')
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	MAIL_LOG_NAME = 'mail.log'
	log_path = log_dir + '/' + MAIL_LOG_NAME
	try:
		# perform mail initialization tasks
		mail.start(args.modeltype, args.modelargs, log_path)
	except Exception as e:
		print('The following error occurred when initializing mail:\n' + str(e) + '\nPlease check ' + log_path + ' for more information')
		sys.exit(1)

	subscriber.subscribe(subscription_path, callback=callback)
	# The subscriber is non-blocking. We must keep the main thread from
	# exiting to allow it to process messages asynchronously in the background.
	print('Listening for messages on {}'.format(subscription_path))
	while True:
		time.sleep(60)
