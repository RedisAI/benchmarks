# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: imagedata.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='imagedata.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0fimagedata.proto\"H\n\tImageData\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\r\n\x05\x64type\x18\x04 \x01(\t\"!\n\x0fPredictionClass\x12\x0e\n\x06output\x18\x01 \x03(\x02\x32<\n\tPredictor\x12/\n\rGetPrediction\x12\n.ImageData\x1a\x10.PredictionClass\"\x00\x62\x06proto3')
)




_IMAGEDATA = _descriptor.Descriptor(
  name='ImageData',
  full_name='ImageData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image', full_name='ImageData.image', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='ImageData.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='ImageData.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='ImageData.dtype', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=91,
)


_PREDICTIONCLASS = _descriptor.Descriptor(
  name='PredictionClass',
  full_name='PredictionClass',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='output', full_name='PredictionClass.output', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=126,
)

DESCRIPTOR.message_types_by_name['ImageData'] = _IMAGEDATA
DESCRIPTOR.message_types_by_name['PredictionClass'] = _PREDICTIONCLASS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImageData = _reflection.GeneratedProtocolMessageType('ImageData', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEDATA,
  __module__ = 'imagedata_pb2'
  # @@protoc_insertion_point(class_scope:ImageData)
  ))
_sym_db.RegisterMessage(ImageData)

PredictionClass = _reflection.GeneratedProtocolMessageType('PredictionClass', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTIONCLASS,
  __module__ = 'imagedata_pb2'
  # @@protoc_insertion_point(class_scope:PredictionClass)
  ))
_sym_db.RegisterMessage(PredictionClass)



_PREDICTOR = _descriptor.ServiceDescriptor(
  name='Predictor',
  full_name='Predictor',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=128,
  serialized_end=188,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetPrediction',
    full_name='Predictor.GetPrediction',
    index=0,
    containing_service=None,
    input_type=_IMAGEDATA,
    output_type=_PREDICTIONCLASS,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_PREDICTOR)

DESCRIPTOR.services_by_name['Predictor'] = _PREDICTOR

# @@protoc_insertion_point(module_scope)
