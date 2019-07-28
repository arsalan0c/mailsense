#!/usr/bin/env python3

import sqlite3
from datetime import datetime, timezone

class metrics(object):
	'''Responsible for the project's statistics.
	'''
	def __init__(self):
		'''Initializes a metrics object.

		Defines database name.
		Defines tables.
		Initializes database objects.
		Creates tables.
		'''
		super(metrics, self).__init__()
		self.DB_NAME = 'mailsense.db'
		# table information store: name as key, list of tuples (field name, field type) as value
		self.TABLES_NAME_FIELDS = {
			'polarities': [('datetime', 'text'), ('polarity', 'text')]
		}
		self.setDBObjects()
		self.createTables()
		self.conn.commit()

	def setDBObjects(self):
		'''Creates new instances of database objects.

		Creates new connection object.
		Creates new cursor object.
		'''
		self.conn = sqlite3.connect(self.DB_NAME)
		self.c = self.conn.cursor()

	def createTables(self):
		'''Creates tables in the database if they don't already exist.
		'''
		for table_name in self.TABLES_NAME_FIELDS.keys():
			create_table_query = "create table if not exists {tn} (".format(tn=table_name)
			# add fields to query
			for field_tuple in self.TABLES_NAME_FIELDS[table_name]:
				create_table_query = create_table_query + '{fn} {ft}, '.format(fn=field_tuple[0], ft=field_tuple[1])
			# remove comma and space, add closing parentheses
			create_table_query = create_table_query[:len(create_table_query)-2] + ")"
			self.c.execute(create_table_query)

	def addPolarity(self, value):
		'''Adds an entry to the 'polarities' table, with the current datetime and a polarity string value.

		Args:
			value: The polarity string value to be added.
		'''
		# to overcome thread-unsafe db objects
		self.setDBObjects()
		self.c.execute('insert into polarities values (?, ?)', (datetime.now(timezone.utc).astimezone(), value))
		self.conn.commit()

	def printPolarityCounts(self):
		'''Prints the total count of the 'polarities' table, along with the count of each polarity value.
		'''
		to_print = '---Polarity Counts---\n'
		for row in self.c.execute('select count(*) from {tn}'.format(tn='polarities')):
			to_print = to_print + 'Total: {tc}\n'.format(tc=row[0])
			break

		for row in self.c.execute('select polarity, count(polarity) from polarities group by polarity'):
			to_print = to_print + '{pv}: {pc}\n'.format(pv=row[0], pc=row[1])

		print(to_print)

if __name__ == '__main__':
	metrics = metrics()
	metrics.printPolarityCounts()
