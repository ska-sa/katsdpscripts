#!/usr/bin/python

#Given a filename this script will list the date and location of 
#reduction products in the archive that have been produced by it.

import optparse
import pysolr
import os

pysolr_database_url='http://kat-archive.kat.ac.za:8983/solr/kat_core'

def get_reduction_metadata(filename, reduction_name=None):
    #Get all reduction products from filename  with given reduction_name 
    #(or all reduction products if reduction_name is None)
    mysolr = pysolr.Solr(pysolr_database_url)
    fn_search_result = mysolr.search('Filename:'+filename)
    if fn_search_result.hits < 1:
        return []
    CASProductId = fn_search_result.docs[0]['CAS.ProductId']
    reduction_products = mysolr.search('InputDataProductId:'+CASProductId)
    if reduction_name==None:
        return reduction_products.docs
    else:
        return [product for product in reduction_products.docs if product.get('ReductionName')==reduction_name]

#command-line parameters
parser = optparse.OptionParser(usage="Please specify the input file\n \
    USAGE: get_reduction_products.py <filename>", \
    description="List reduction production associated with input filename")
opts, args = parser.parse_args()

if len(args)==1:
	input_file=os.path.basename(args[0])
else:
	parser.error("incorrect number of arguments")

if input_file is '':
	parser.error("no filename specified")

output=get_reduction_metadata(input_file)

print "Reductions in archive for ", input_file + ":\n"
print "%-40s  %-20s   %-56s"%("Reduction Name:","Reduction Date:", "Archive Location:",)
print "%-40s  %-20s   %-56s"%("---------------","---------------", "-----------------",)
for result in output:
	reduction_name=result['ReductionName']
	file_location=os.path.join(result['FileLocation'][0],result['ProductName'])
	reduction_date=result['StartTime']
	print "%-40s  %-20s   %-56s"%(reduction_name,reduction_date,file_location,)
