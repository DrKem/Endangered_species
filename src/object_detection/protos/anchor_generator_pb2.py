# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/anchor_generator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)
import object_detection

_sym_db = _symbol_database.Default()

from object_detection.protos import grid_anchor_generator_pb2
from object_detection.protos import ssd_anchor_generator_pb2 
#import object_detection.protos.grid_anchor_generator_pb2
#import object_detection.protos.ssd_anchor_generator_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/anchor_generator.proto',
  package='object_detection.protos',
  serialized_pb=_b('\n.object_detection/protos/anchor_generator.proto\x12\x17object_detection.protos\x1a\x33object_detection/protos/grid_anchor_generator.proto\x1a\x32object_detection/protos/ssd_anchor_generator.proto\"\xc7\x01\n\x0f\x41nchorGenerator\x12M\n\x15grid_anchor_generator\x18\x01 \x01(\x0b\x32,.object_detection.protos.GridAnchorGeneratorH\x00\x12K\n\x14ssd_anchor_generator\x18\x02 \x01(\x0b\x32+.object_detection.protos.SsdAnchorGeneratorH\x00\x42\x18\n\x16\x61nchor_generator_oneof')
  ,
  dependencies=[object_detection.protos.grid_anchor_generator_pb2.DESCRIPTOR,object_detection.protos.ssd_anchor_generator_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_ANCHORGENERATOR = _descriptor.Descriptor(
  name='AnchorGenerator',
  full_name='object_detection.protos.AnchorGenerator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='grid_anchor_generator', full_name='object_detection.protos.AnchorGenerator.grid_anchor_generator', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ssd_anchor_generator', full_name='object_detection.protos.AnchorGenerator.ssd_anchor_generator', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='anchor_generator_oneof', full_name='object_detection.protos.AnchorGenerator.anchor_generator_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=181,
  serialized_end=380,
)

_ANCHORGENERATOR.fields_by_name['grid_anchor_generator'].message_type = object_detection.protos.grid_anchor_generator_pb2._GRIDANCHORGENERATOR
_ANCHORGENERATOR.fields_by_name['ssd_anchor_generator'].message_type = object_detection.protos.ssd_anchor_generator_pb2._SSDANCHORGENERATOR
_ANCHORGENERATOR.oneofs_by_name['anchor_generator_oneof'].fields.append(
  _ANCHORGENERATOR.fields_by_name['grid_anchor_generator'])
_ANCHORGENERATOR.fields_by_name['grid_anchor_generator'].containing_oneof = _ANCHORGENERATOR.oneofs_by_name['anchor_generator_oneof']
_ANCHORGENERATOR.oneofs_by_name['anchor_generator_oneof'].fields.append(
  _ANCHORGENERATOR.fields_by_name['ssd_anchor_generator'])
_ANCHORGENERATOR.fields_by_name['ssd_anchor_generator'].containing_oneof = _ANCHORGENERATOR.oneofs_by_name['anchor_generator_oneof']
DESCRIPTOR.message_types_by_name['AnchorGenerator'] = _ANCHORGENERATOR

AnchorGenerator = _reflection.GeneratedProtocolMessageType('AnchorGenerator', (_message.Message,), dict(
  DESCRIPTOR = _ANCHORGENERATOR,
  __module__ = 'object_detection.protos.anchor_generator_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.AnchorGenerator)
  ))
_sym_db.RegisterMessage(AnchorGenerator)


# @@protoc_insertion_point(module_scope)
