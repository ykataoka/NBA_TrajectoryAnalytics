#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xmltodict
import json
import glob

# xml = '''<?xml version="1.0"?>
# <root>
#   <parent>
#     <child a="foo">
#       Hello, world!!
#     </child>
#   </parent>
# </root>
# '''

xmlfiles = glob.glob('*.XML')
xsdfiles = glob.glob('*.xsd')

print xmlfiles
print xsdfiles

if __name__ == '__main__':
    for filename in xmlfiles:
        str = open(filename, 'r').read()
        result = xmltodict.parse(str)
        print(json.dumps(result, indent=2))
        print filename
        raw_input()

    for filename in xsdfiles:
        str = open(filename, 'r').read()
        result = xmltodict.parse(str)
        print(json.dumps(result, indent=2))
        print filename
        raw_input()

        #    print(result)
