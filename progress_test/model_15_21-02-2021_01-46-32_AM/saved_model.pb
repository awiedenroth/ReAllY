ох
њ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02unknown8Г∞
Т
my_model_37/dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namemy_model_37/dense_74/kernel
Л
/my_model_37/dense_74/kernel/Read/ReadVariableOpReadVariableOpmy_model_37/dense_74/kernel*
_output_shapes

:*
dtype0
К
my_model_37/dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namemy_model_37/dense_74/bias
Г
-my_model_37/dense_74/bias/Read/ReadVariableOpReadVariableOpmy_model_37/dense_74/bias*
_output_shapes
:*
dtype0
Т
my_model_37/dense_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namemy_model_37/dense_75/kernel
Л
/my_model_37/dense_75/kernel/Read/ReadVariableOpReadVariableOpmy_model_37/dense_75/kernel*
_output_shapes

:*
dtype0
К
my_model_37/dense_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namemy_model_37/dense_75/bias
Г
-my_model_37/dense_75/bias/Read/ReadVariableOpReadVariableOpmy_model_37/dense_75/bias*
_output_shapes
:*
dtype0

NoOpNoOp
 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Е
valueы
Bш
 Bс

y
	layer

layer2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
	bias

trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
	1
2
3

0
	1
2
3
 
≠
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
 
XV
VARIABLE_VALUEmy_model_37/dense_74/kernel'layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_model_37/dense_74/bias%layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
	1

0
	1
 
≠

trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
YW
VARIABLE_VALUEmy_model_37/dense_75/kernel(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmy_model_37/dense_75/bias&layer2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
 layer_metrics
!layer_regularization_losses
"metrics

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Њ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_37/dense_74/kernelmy_model_37/dense_74/biasmy_model_37/dense_75/kernelmy_model_37/dense_75/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_462568
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
я
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/my_model_37/dense_74/kernel/Read/ReadVariableOp-my_model_37/dense_74/bias/Read/ReadVariableOp/my_model_37/dense_75/kernel/Read/ReadVariableOp-my_model_37/dense_75/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_462642
К
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_37/dense_74/kernelmy_model_37/dense_74/biasmy_model_37/dense_75/kernelmy_model_37/dense_75/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_462664іП
№
~
)__inference_dense_74_layer_call_fn_462587

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_4624912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
л
д
"__inference__traced_restore_462664
file_prefix0
,assignvariableop_my_model_37_dense_74_kernel0
,assignvariableop_1_my_model_37_dense_74_bias2
.assignvariableop_2_my_model_37_dense_75_kernel0
,assignvariableop_3_my_model_37_dense_75_bias

identity_5ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3…
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*’
valueЋB»B'layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB%layer/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesШ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesƒ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЂ
AssignVariableOpAssignVariableOp,assignvariableop_my_model_37_dense_74_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_my_model_37_dense_74_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2≥
AssignVariableOp_2AssignVariableOp.assignvariableop_2_my_model_37_dense_75_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_my_model_37_dense_75_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4ђ

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
В	
∞
,__inference_my_model_37_layer_call_fn_462551
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1ИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_my_model_37_layer_call_and_return_conditional_losses_4625352
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ќ
ђ
D__inference_dense_74_layer_call_and_return_conditional_losses_462491

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ
ђ
D__inference_dense_75_layer_call_and_return_conditional_losses_462597

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ
ђ
D__inference_dense_75_layer_call_and_return_conditional_losses_462517

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘
®
$__inference_signature_wrapper_462568
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1ИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_4624772
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
№
~
)__inference_dense_75_layer_call_fn_462606

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_4625172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ќ
С
G__inference_my_model_37_layer_call_and_return_conditional_losses_462535
input_1
dense_74_462502
dense_74_462504
dense_75_462528
dense_75_462530
identity

identity_1ИҐ dense_74/StatefulPartitionedCallҐ dense_75/StatefulPartitionedCallХ
 dense_74/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_74_462502dense_74_462504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_4624912"
 dense_74/StatefulPartitionedCallЈ
 dense_75/StatefulPartitionedCallStatefulPartitionedCall)dense_74/StatefulPartitionedCall:output:0dense_75_462528dense_75_462530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_75_layer_call_and_return_conditional_losses_4625172"
 dense_75/StatefulPartitionedCall√
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity«

Identity_1Identity)dense_75/StatefulPartitionedCall:output:0!^dense_74/StatefulPartitionedCall!^dense_75/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€::::2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2D
 dense_75/StatefulPartitionedCall dense_75/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
ф
Ј
!__inference__wrapped_model_462477
input_17
3my_model_37_dense_74_matmul_readvariableop_resource8
4my_model_37_dense_74_biasadd_readvariableop_resource7
3my_model_37_dense_75_matmul_readvariableop_resource8
4my_model_37_dense_75_biasadd_readvariableop_resource
identity

identity_1Ић
*my_model_37/dense_74/MatMul/ReadVariableOpReadVariableOp3my_model_37_dense_74_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*my_model_37/dense_74/MatMul/ReadVariableOp≥
my_model_37/dense_74/MatMulMatMulinput_12my_model_37/dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_37/dense_74/MatMulЋ
+my_model_37/dense_74/BiasAdd/ReadVariableOpReadVariableOp4my_model_37_dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+my_model_37/dense_74/BiasAdd/ReadVariableOp’
my_model_37/dense_74/BiasAddBiasAdd%my_model_37/dense_74/MatMul:product:03my_model_37/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_37/dense_74/BiasAddћ
*my_model_37/dense_75/MatMul/ReadVariableOpReadVariableOp3my_model_37_dense_75_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*my_model_37/dense_75/MatMul/ReadVariableOp—
my_model_37/dense_75/MatMulMatMul%my_model_37/dense_74/BiasAdd:output:02my_model_37/dense_75/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_37/dense_75/MatMulЋ
+my_model_37/dense_75/BiasAdd/ReadVariableOpReadVariableOp4my_model_37_dense_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+my_model_37/dense_75/BiasAdd/ReadVariableOp’
my_model_37/dense_75/BiasAddBiasAdd%my_model_37/dense_75/MatMul:product:03my_model_37/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_37/dense_75/BiasAddy
IdentityIdentity%my_model_37/dense_74/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity}

Identity_1Identity%my_model_37/dense_75/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€:::::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ќ
ђ
D__inference_dense_74_layer_call_and_return_conditional_losses_462578

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
Ў
__inference__traced_save_462642
file_prefix:
6savev2_my_model_37_dense_74_kernel_read_readvariableop8
4savev2_my_model_37_dense_74_bias_read_readvariableop:
6savev2_my_model_37_dense_75_kernel_read_readvariableop8
4savev2_my_model_37_dense_75_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1dba0688a8024c398ea9011d7b09b28b/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename√
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*’
valueЋB»B'layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB%layer/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesЪ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_my_model_37_dense_74_kernel_read_readvariableop4savev2_my_model_37_dense_74_bias_read_readvariableop6savev2_my_model_37_dense_75_kernel_read_readvariableop4savev2_my_model_37_dense_75_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: "ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*п
serving_defaultџ
;
input_10
serving_default_input_1:0€€€€€€€€€<
q_values0
StatefulPartitionedCall:0€€€€€€€€€B
value_estimate0
StatefulPartitionedCall:1€€€€€€€€€tensorflow/serving/predict:Б>
ќ
	layer

layer2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
#__call__
$_default_save_signature
*%&call_and_return_all_conditional_losses"ы
_tf_keras_modelб{"class_name": "MyModel", "name": "my_model_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}}
о

kernel
	bias

trainable_variables
	variables
regularization_losses
	keras_api
&__call__
*'&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "Dense", "name": "dense_74", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_74", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
о

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "Dense", "name": "dense_75", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_75", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
<
0
	1
2
3"
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
#__call__
$_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
-:+2my_model_37/dense_74/kernel
':%2my_model_37/dense_74/bias
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠

trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
-:+2my_model_37/dense_75/kernel
':%2my_model_37/dense_75/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
 layer_metrics
!layer_regularization_losses
"metrics
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ш2х
,__inference_my_model_37_layer_call_fn_462551ƒ
Ч≤У
FullArgSpec
argsЪ
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
я2№
!__inference__wrapped_model_462477ґ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
У2Р
G__inference_my_model_37_layer_call_and_return_conditional_losses_462535ƒ
Ч≤У
FullArgSpec
argsЪ
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
”2–
)__inference_dense_74_layer_call_fn_462587Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_74_layer_call_and_return_conditional_losses_462578Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_75_layer_call_fn_462606Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_75_layer_call_and_return_conditional_losses_462597Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
3B1
$__inference_signature_wrapper_462568input_1ѕ
!__inference__wrapped_model_462477©	0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "o™l
.
q_values"К
q_values€€€€€€€€€
:
value_estimate(К%
value_estimate€€€€€€€€€§
D__inference_dense_74_layer_call_and_return_conditional_losses_462578\	/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_74_layer_call_fn_462587O	/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€§
D__inference_dense_75_layer_call_and_return_conditional_losses_462597\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_75_layer_call_fn_462606O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Г
G__inference_my_model_37_layer_call_and_return_conditional_losses_462535Ј	0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "}Ґz
s™p
0
q_values$К!

0/q_values€€€€€€€€€
<
value_estimate*К'
0/value_estimate€€€€€€€€€
Ъ Џ
,__inference_my_model_37_layer_call_fn_462551©	0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "o™l
.
q_values"К
q_values€€€€€€€€€
:
value_estimate(К%
value_estimate€€€€€€€€€Ё
$__inference_signature_wrapper_462568і	;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"o™l
.
q_values"К
q_values€€€€€€€€€
:
value_estimate(К%
value_estimate€€€€€€€€€