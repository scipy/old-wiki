""" Modified version of array_converter that adds a type definition macro
   for each array.
"""

from scipy.weave.standard_array_spec import array_converter
from scipy.weave import c_spec

##############################################################################
# Weave modifications
##############################################################################

class typed_array_converter(array_converter):
    """ Minor change to the original array type converter that adds a macro
        for the array's data type.
    """
    
    def declaration_code(self, templatize = 0,inline=0):
        code = array_converter.declaration_code(self, templatize, 
                                                inline)
        res = self.template_vars(inline=inline)
        # need to add a macro that defines the array's data type
        code += '#define %(name)s_data_type %(num_type)s\n' % res
        
        return code

# Create a list of type converters that includes this array converter instead
# of the standard one.
converters = [c_spec.int_converter(),
              c_spec.float_converter(),
              c_spec.complex_converter(),
              c_spec.unicode_converter(),
              c_spec.string_converter(),
              c_spec.list_converter(),
              c_spec.dict_converter(),
              c_spec.tuple_converter(),
              c_spec.file_converter(),
              c_spec.instance_converter(),
              typed_array_converter()]


##############################################################################
