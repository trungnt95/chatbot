# This Python file uses the following encoding: utf-8
import sys
def levenshtein_distance(statement, other_statement):
	import sys
	from difflib import SequenceMatcher
	
	if not statement or not other_statement:
		return 0
		
	statement_text = statement.lower()
	other_statement_text = other_statement.lower()
	
	similarity = SequenceMatcher(
		None,
		statement_text,
		other_statement_text
	)
	
	percent = round(similarity.ratio(), 2)
	return percent


