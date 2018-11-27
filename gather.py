import csv
import getopt
import json
import sys
import urllib.request

def main(argv):
	inputyear = ""
	first = True
	try:
		opts, args = getopt.getopt(argv,"y:",["year="])
	except getopt.GetoptError:
		print('gather.py -y <year>')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-y", "--year"):
			inputyear = arg

	inputyear = int(inputyear)

	with urllib.request.urlopen('http://www.nhl.com/stats/rest/team?isAggregate=false&reportType=basic&isGame=false&reportName=teamsummary&sort=[{{%22property%22:%22teamId%22,%22direction%22:%22ASC%22}},{{%22property%22:%22wins%22,%22direction%22:%22DESC%22}}]&cayenneExp=gameTypeId=2%20and%20seasonId%3E={0}%20and%20seasonId%3C={0}'.format(str(inputyear)+str(inputyear+1))) as url:
		data = json.loads(url.read().decode())
	
	with open(str(inputyear)+'nhlstats.csv', 'a') as csvfile:
		if first:
			fieldnames = []
			for field in data['data'][0]:
				fieldnames.append(field)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
		if first:
			writer.writeheader()
			first = False
		for row in data['data']:
			writer.writerow(row)
			
	first = True
	for year in range(1999,inputyear):

		with urllib.request.urlopen("http://www.nhl.com/stats/rest/team?isAggregate=false&reportType=basic&isGame=false&reportName=teamsummary&sort=[{{%22property%22:%22teamId%22,%22direction%22:%22ASC%22}},{{%22property%22:%22wins%22,%22direction%22:%22DESC%22}}]&cayenneExp=gameTypeId=2%20and%20seasonId%3E={0}%20and%20seasonId%3C={0}".format(str(year)+str(year+1))) as url:
			data = json.loads(url.read().decode())
		
		with urllib.request.urlopen("http://www.nhl.com/stats/rest/team?isAggregate=false&reportType=basic&isGame=false&reportName=teamsummary&sort=[{{%22property%22:%22teamId%22,%22direction%22:%22ASC%22}},{{%22property%22:%22wins%22,%22direction%22:%22DESC%22}}]&cayenneExp=gameTypeId=3%20and%20seasonId%3E={0}%20and%20seasonId%3C={0}".format(str(year)+str(year+1))) as url:
			playoffdata = json.loads(url.read().decode())
		
		with open('allnhlstats.csv', 'a') as csvfile:
			if first:
				fieldnames = []
				for field in data['data'][0]:
					fieldnames.append(field)
				fieldnames.append('playoffwins')
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
			if first:
				writer.writeheader()
				first = False
			for playoffrow in playoffdata['data']:
				for rownum in range(len(data['data'])):
					if data['data'][rownum-1]['teamAbbrev'] == playoffrow['teamAbbrev']:
						data['data'][rownum-1]['playoffwins'] = playoffrow['wins']
						writer.writerow(data['data'][rownum-1])
				
if __name__ == "__main__":
   main(sys.argv[1:])