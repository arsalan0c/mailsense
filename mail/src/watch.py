#!/usr/bin/env python3

# This script calls watch in order to receive push notifications for inbox changes from Gmail.
# It must be called every 7 days in order for the notifications to continue to be received.
# Gmail documentation recommends calling it daily.

def watch(service, project, topic):
	'''Calls watch on a Gmail inbox

	Args:
		service: A Gmail service object to access the Gmail API.
		project: Name of the project from Google Cloud
		topic: Name of the topic from Google Cloud
	'''
	request = {
	    'labelIds': ['INBOX'],
	    'topicName': 'projects/' + project + '/topics/' + topic,
	}

	# get the Gmail api client
	service.users().watch(userId='me', body=request).execute()

if __name__ == '__main__':
	import argparse
	from mail import get_service

	parser = argparse.ArgumentParser(description='process arguments required for enabling Gmail push notifications')
	parser.add_argument('-p', '--project', help='string: name of the project from Google Cloud', type=str, action='store', required=True)
	parser.add_argument('-t', '--topic', help='string: name of the topic from Google Cloud', type=str, action='store', required=True)
	parser.add_argument('-cp', '--credentialspath', help='string: path to Gmail API credentials file', type=str, action='store', required=True)
	parser.add_argument('-tp', '--tokenpath', help='string: path to Gmail API token file', type=str, action='store', required=True)
	args = parser.parse_args()

	service = get_service(args.credentialspath, aegs.tokenpath)
	watch(service, args.project, args.topic)
