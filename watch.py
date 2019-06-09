#!/usr/bin/env python3

# This script calls watch in order to receive push notifications for inbox changes from Gmail.
# It must be called every 7 days in order for the notifications to continue to be received.
# Gmail documentation recommends calling it daily.

import mail
import argparse

parser = argparse.ArgumentParser(description='process arguments required for enabling Gmail push notifications')
parser.add_argument('-p', '--project', help='string: project name from Google cloud', type=str, action='store', required=True)
parser.add_argument('-t', '--topic', help='string: topic name', type=str, action='store', required=True)
parser.add_argument('-cp', '--credentialspath', help='string: path to Gmail API credentials file', type=str, action='store', required=True)
parser.add_argument('-tp', '--tokenpath', help='string: path to Gmail API token file', type=str, action='store', required=True)
args = parser.parse_args()

request = {
    'labelIds': ['INBOX'],
    'topicName': 'projects/' + args.project + '/topics/' + args.topic,
}

# get the Gmail api client
service = mail.get_service(args.credentialspath, args.tokenpath)

service.users().watch(userId='me', body=request).execute()