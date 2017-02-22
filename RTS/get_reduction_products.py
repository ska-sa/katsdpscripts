#!/usr/bin/python

#Given a filename this script will list the location of reduction products 
#in the archive that have been produced by it.

import optparse
import pysolr

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
    USAGE: python get_reduction_products.py <filename>", \
    description="List reduction production associated with input filename")
opts, args = parser.parse_args()

input_file=args[0]
output=get_reduction_metadata(args[0])

print "Reductions in archive for ", input_file 
print "----------------------------------------"

for result in output:
	reduction_name=result['ReductionName']
	file_location=result['FileLocation'][0]+'/'+result['ProductName']
	print reduction_name,':\t\t',file_location

