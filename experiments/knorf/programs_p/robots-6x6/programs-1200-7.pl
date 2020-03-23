p441(A,B):-move_left(A,C),move_left(C,B).
p715(A,B):-move_right(A,C),move_right(C,B).
p740(A,B):-move_backwards(A,B).
p890(A,B):-move_left(A,C),move_backwards(C,B).
p918(A,B):-move_right(A,B).
p988(A,B):-move_forwards(A,B).
p1120(A,B):-move_backwards(A,C),move_backwards(C,B).
p1197(A,B):-move_forwards(A,C),move_forwards(C,B).
p8(A,B):-move_forwards(A,C),p441(C,B).
p149(A,B):-p441(A,C),p441(C,B).
p198(A,B):-move_left(A,C),p198_1(C,B).
p198_1(A,B):-p890(A,C),p1120(C,B).
p210(A,B):-p441(A,C),p210_1(C,B).
p210_1(A,B):-p441(A,C),p890(C,B).
p260(A,B):-p441(A,C),p260_1(C,B).
p260_1(A,B):-drop_ball(A,C),move_left(C,B).
p261(A,B):-move_left(A,C),p1197(C,B).
p291(A,B):-move_right(A,C),p1197(C,B).
p416(A,B):-move_left(A,C),p890(C,B).
p438(A,B):-move_backwards(A,C),p438_1(C,B).
p438_1(A,B):-p715(A,C),p1120(C,B).
p450(A,B):-move_left(A,C),p450_1(C,B).
p450_1(A,B):-grab_ball(A,C),p1120(C,B).
p497(A,B):-move_right(A,C),p497_1(C,B).
p497_1(A,B):-move_forwards(A,C),p1197(C,B).
p566(A,B):-move_forwards(A,C),p566_1(C,B).
p566_1(A,B):-p715(A,C),p1197(C,B).
p676(A,B):-p441(A,C),p441(C,B).
p751(A,B):-move_right(A,C),p751_1(C,B).
p751_1(A,B):-move_backwards(A,C),p715(C,B).
p823(A,B):-move_left(A,C),p823_1(C,B).
p823_1(A,B):-drop_ball(A,C),move_right(C,B).
p883(A,B):-move_left(A,C),p1120(C,B).
p945(A,B):-move_right(A,C),p945_1(C,B).
p945_1(A,B):-p715(A,C),p1120(C,B).
p1068(A,B):-move_left(A,C),p1068_1(C,B).
p1068_1(A,B):-move_forwards(A,C),p1197(C,B).
p1099(A,B):-move_forwards(A,C),p1099_1(C,B).
p1099_1(A,B):-p715(A,C),p715(C,B).
p38(A,B):-p441(A,C),p38_1(C,B).
p38_1(A,B):-p1068(A,C),p38_2(C,B).
p38_2(A,B):-p450(A,C),move_left(C,B).
p44(A,B):-p566(A,C),p44_1(C,B).
p44_1(A,B):-p450_1(A,C),p44_2(C,B).
p44_2(A,B):-drop_ball(A,C),p198_1(C,B).
p103(A,B):-p291(A,C),p103_1(C,B).
p103_1(A,B):-grab_ball(A,C),p103_2(C,B).
p103_2(A,B):-move_left(A,C),p441(C,B).
p112(A,B):-p883(A,C),p112_1(C,B).
p112_1(A,B):-p450(A,C),p112_2(C,B).
p112_2(A,B):-p1068(A,C),p823(C,B).
p142(A,B):-p1197(A,C),p142_1(C,B).
p142_1(A,B):-p450(A,C),p142_2(C,B).
p142_2(A,B):-p260(A,C),p1099_1(C,B).
p146(A,B):-move_left(A,C),p146_1(C,B).
p146_1(A,B):-move_forwards(A,C),p146_2(C,B).
p146_2(A,B):-grab_ball(A,C),p751_1(C,B).
p173(A,B):-p8(A,C),p173_1(C,B).
p173_1(A,B):-grab_ball(A,C),p173_2(C,B).
p173_2(A,B):-p260(A,C),p566_1(C,B).
p176(A,B):-move_right(A,C),p176_1(C,B).
p176_1(A,B):-grab_ball(A,C),p176_2(C,B).
p176_2(A,B):-p260(A,C),move_backwards(C,B).
p196(A,B):-move_forwards(A,C),p196_1(C,B).
p196_1(A,B):-p715(A,C),p196_2(C,B).
p196_2(A,B):-drop_ball(A,C),p210_1(C,B).
p205(A,B):-move_left(A,C),p205_1(C,B).
p205_1(A,B):-grab_ball(A,C),p205_2(C,B).
p205_2(A,B):-p1197(A,C),drop_ball(C,B).
p206(A,B):-p441(A,C),p206_1(C,B).
p206_1(A,B):-grab_ball(A,C),p206_2(C,B).
p206_2(A,B):-p1099(A,C),p823_1(C,B).
p231(A,B):-move_right(A,C),p231_1(C,B).
p231_1(A,B):-move_forwards(A,C),p823_1(C,B).
p240(A,B):-move_backwards(A,C),p240_1(C,B).
p240_1(A,B):-p450(A,C),p240_2(C,B).
p240_2(A,B):-p823(A,C),p1197(C,B).
p241(A,B):-p450(A,C),p241_1(C,B).
p241_1(A,B):-p438_1(A,C),p241_2(C,B).
p241_2(A,B):-p260_1(A,C),p261(C,B).
p242(A,B):-p450_1(A,C),p242_1(C,B).
p242_1(A,B):-p438(A,C),p242_2(C,B).
p242_2(A,B):-p260(A,C),p497(C,B).
p253(A,B):-move_forwards(A,C),p253_1(C,B).
p253_1(A,B):-p450_1(A,C),p253_2(C,B).
p253_2(A,B):-p823(A,C),p1068(C,B).
p281(A,B):-p291(A,C),p281_1(C,B).
p281_1(A,B):-p450_1(A,C),p281_2(C,B).
p281_2(A,B):-p260(A,C),p1099(C,B).
p299(A,B):-move_forwards(A,C),p299_1(C,B).
p299_1(A,B):-p497_1(A,C),p299_2(C,B).
p299_2(A,B):-drop_ball(A,C),p8(C,B).
p310(A,B):-p438(A,C),p450_1(C,B).
p329(A,B):-p438(A,C),p329_1(C,B).
p329_1(A,B):-grab_ball(A,C),p329_2(C,B).
p329_2(A,B):-p260(A,C),p1099_1(C,B).
p333(A,B):-p890(A,C),p333_1(C,B).
p333_1(A,B):-p438(A,C),p333_2(C,B).
p333_2(A,B):-p260_1(A,C),p1099(C,B).
p337(A,B):-p751(A,C),p337_1(C,B).
p337_1(A,B):-grab_ball(A,C),p337_2(C,B).
p337_2(A,B):-p260(A,C),p497(C,B).
p340(A,B):-p1099(A,C),p340_1(C,B).
p340_1(A,B):-p450_1(A,C),p441(C,B).
p342(A,B):-p1099_1(A,C),p342_1(C,B).
p342_1(A,B):-grab_ball(A,C),p342_2(C,B).
p342_2(A,B):-p149(A,C),drop_ball(C,B).
p363(A,B):-p715(A,C),p363_1(C,B).
p363_1(A,B):-p450_1(A,C),p363_2(C,B).
p363_2(A,B):-drop_ball(A,C),p291(C,B).
p385(A,B):-move_forwards(A,C),p385_1(C,B).
p385_1(A,B):-p450(A,C),p385_2(C,B).
p385_2(A,B):-move_right(A,C),p566_1(C,B).
p393(A,B):-p890(A,C),p393_1(C,B).
p393_1(A,B):-p260(A,C),p393_2(C,B).
p393_2(A,B):-p438_1(A,C),p1099_1(C,B).
p394(A,B):-p416(A,C),p394_1(C,B).
p394_1(A,B):-grab_ball(A,C),p1197(C,B).
p397(A,B):-p1197(A,C),p397_1(C,B).
p397_1(A,B):-p823(A,C),p438(C,B).
p415(A,B):-p415_1(A,C),p415_1(C,B).
p415_1(A,B):-move_right(A,C),p415_2(C,B).
p415_2(A,B):-p450_1(A,C),p260(C,B).
p429(A,B):-p1099(A,C),p429_1(C,B).
p429_1(A,B):-p450_1(A,C),p429_2(C,B).
p429_2(A,B):-p260(A,C),p1068(C,B).
p436(A,B):-p497(A,C),p436_1(C,B).
p436_1(A,B):-grab_ball(A,C),p436_2(C,B).
p436_2(A,B):-p198_1(A,C),p260(C,B).
p459(A,B):-p450(A,C),p459_1(C,B).
p459_1(A,B):-p8(A,C),p459_2(C,B).
p459_2(A,B):-drop_ball(A,C),p715(C,B).
p471(A,B):-p441(A,C),p1068(C,B).
p475(A,B):-grab_ball(A,C),p475_1(C,B).
p475_1(A,B):-p438_1(A,C),p475_2(C,B).
p475_2(A,B):-p823(A,C),move_forwards(C,B).
p481(A,B):-p751_1(A,C),p481_1(C,B).
p481_1(A,B):-p450(A,C),p481_2(C,B).
p481_2(A,B):-p291(A,C),p823_1(C,B).
p499(A,B):-grab_ball(A,C),p499_1(C,B).
p499_1(A,B):-move_right(A,C),p499_2(C,B).
p499_2(A,B):-drop_ball(A,C),p261(C,B).
p500(A,B):-p1068(A,C),p500_1(C,B).
p500_1(A,B):-p450(A,C),p500_2(C,B).
p500_2(A,B):-p260(A,C),move_backwards(C,B).
p504(A,B):-move_right(A,C),p504_1(C,B).
p504_1(A,B):-p450_1(A,C),p504_2(C,B).
p504_2(A,B):-p823(A,C),p566_1(C,B).
p517(A,B):-p751_1(A,C),p517_1(C,B).
p517_1(A,B):-grab_ball(A,C),p517_2(C,B).
p517_2(A,B):-p1197(A,C),p149(C,B).
p534(A,B):-move_backwards(A,C),p534_1(C,B).
p534_1(A,B):-grab_ball(A,C),p534_2(C,B).
p534_2(A,B):-p1099(A,C),p260(C,B).
p555(A,B):-move_left(A,C),p555_1(C,B).
p555_1(A,B):-p751_1(A,C),p555_2(C,B).
p555_2(A,B):-grab_ball(A,C),p8(C,B).
p569(A,B):-p751_1(A,C),p569_1(C,B).
p569_1(A,B):-p450(A,C),p569_2(C,B).
p569_2(A,B):-p497(A,C),p260_1(C,B).
p589(A,B):-p441(A,C),p589_1(C,B).
p589_1(A,B):-p450(A,C),p589_2(C,B).
p589_2(A,B):-move_left(A,C),p260_1(C,B).
p598(A,B):-p883(A,C),p598_1(C,B).
p598_1(A,B):-p450(A,C),p598_2(C,B).
p598_2(A,B):-p291(A,C),p823_1(C,B).
p646(A,B):-move_forwards(A,C),p646_1(C,B).
p646_1(A,B):-p450_1(A,C),p646_2(C,B).
p646_2(A,B):-p1099(A,C),p260_1(C,B).
p647(A,B):-p1099_1(A,C),p647_1(C,B).
p647_1(A,B):-grab_ball(A,C),p647_2(C,B).
p647_2(A,B):-p8(A,C),p823_1(C,B).
p652(A,B):-p497_1(A,C),p652_1(C,B).
p652_1(A,B):-p450_1(A,C),p652_2(C,B).
p652_2(A,B):-drop_ball(A,C),p566_1(C,B).
p675(A,B):-p1197(A,C),p566(C,B).
p683(A,B):-p8(A,C),p683_1(C,B).
p683_1(A,B):-grab_ball(A,C),p683_2(C,B).
p683_2(A,B):-p438_1(A,C),p823(C,B).
p688(A,B):-p450(A,C),p688_1(C,B).
p688_1(A,B):-move_backwards(A,C),p688_2(C,B).
p688_2(A,B):-drop_ball(A,C),p1068(C,B).
p711(A,B):-p1197(A,C),p210(C,B).
p726(A,B):-p8(A,C),p726_1(C,B).
p726_1(A,B):-p260(A,C),move_backwards(C,B).
p736(A,B):-move_backwards(A,C),p736_1(C,B).
p736_1(A,B):-p450(A,C),p736_2(C,B).
p736_2(A,B):-drop_ball(A,C),p497_1(C,B).
p749(A,B):-p438_1(A,C),p749_1(C,B).
p749_1(A,B):-p751(A,C),p749_2(C,B).
p749_2(A,B):-grab_ball(A,C),p890(C,B).
p752(A,B):-p751(A,C),p752_1(C,B).
p752_1(A,B):-grab_ball(A,C),move_left(C,B).
p755(A,B):-p566(A,C),p755_1(C,B).
p755_1(A,B):-drop_ball(A,C),p261(C,B).
p766(A,B):-p1197(A,C),p766_1(C,B).
p766_1(A,B):-p450(A,C),p766_2(C,B).
p766_2(A,B):-p1099(A,C),drop_ball(C,B).
p773(A,B):-grab_ball(A,C),p773_1(C,B).
p773_1(A,B):-move_backwards(A,C),p773_2(C,B).
p773_2(A,B):-drop_ball(A,C),p751_1(C,B).
p784(A,B):-p198_1(A,C),p784_1(C,B).
p784_1(A,B):-grab_ball(A,C),p784_2(C,B).
p784_2(A,B):-move_right(A,C),p566(C,B).
p785(A,B):-p450(A,C),p785_1(C,B).
p785_1(A,B):-p261(A,C),p785_2(C,B).
p785_2(A,B):-drop_ball(A,C),p438(C,B).
p811(A,B):-p291(A,C),p566_1(C,B).
p850(A,B):-p883(A,C),p850_1(C,B).
p850_1(A,B):-p450(A,C),p850_2(C,B).
p850_2(A,B):-p566(A,C),p260_1(C,B).
p853(A,B):-p450(A,C),p853_1(C,B).
p853_1(A,B):-p823(A,C),p853_2(C,B).
p853_2(A,B):-move_forwards(A,C),p497_1(C,B).
p855(A,B):-p1197(A,C),p855_1(C,B).
p855_1(A,B):-p149(A,C),p1068(C,B).
p874(A,B):-p1197(A,C),p874_1(C,B).
p874_1(A,B):-grab_ball(A,C),p874_2(C,B).
p874_2(A,B):-move_backwards(A,C),p210(C,B).
p893(A,B):-p1068(A,C),p893_1(C,B).
p893_1(A,B):-p450(A,C),p893_2(C,B).
p893_2(A,B):-drop_ball(A,C),move_forwards(C,B).
p899(A,B):-p497_1(A,C),p899_1(C,B).
p899_1(A,B):-p450(A,C),p823_1(C,B).
p939(A,B):-p1099_1(A,C),p939_1(C,B).
p939_1(A,B):-grab_ball(A,C),p939_2(C,B).
p939_2(A,B):-p198(A,C),p823(C,B).
p942(A,B):-p890(A,C),p942_1(C,B).
p942_1(A,B):-grab_ball(A,C),p942_2(C,B).
p942_2(A,B):-p1197(A,C),p823_1(C,B).
p946(A,B):-p1197(A,C),p946_1(C,B).
p946_1(A,B):-grab_ball(A,C),p946_2(C,B).
p946_2(A,B):-p438_1(A,C),p260_1(C,B).
p980(A,B):-p1197(A,C),p980_1(C,B).
p980_1(A,B):-p450(A,C),p980_2(C,B).
p980_2(A,B):-move_right(A,C),p823_1(C,B).
p999(A,B):-p8(A,C),p999_1(C,B).
p999_1(A,B):-p450(A,C),p999_2(C,B).
p999_2(A,B):-drop_ball(A,C),p751_1(C,B).
p1022(A,B):-grab_ball(A,C),p1022_1(C,B).
p1022_1(A,B):-p260(A,C),p1099_1(C,B).
p1027(A,B):-move_forwards(A,C),p1027_1(C,B).
p1027_1(A,B):-p450(A,C),p1027_2(C,B).
p1027_2(A,B):-drop_ball(A,C),p566(C,B).
p1038(A,B):-p497_1(A,C),p1038_1(C,B).
p1038_1(A,B):-grab_ball(A,C),move_forwards(C,B).
p1041(A,B):-p8(A,C),p1041_1(C,B).
p1041_1(A,B):-p450(A,C),p1041_2(C,B).
p1041_2(A,B):-p261(A,C),p260_1(C,B).
p1043(A,B):-p441(A,C),p1043_1(C,B).
p1043_1(A,B):-p450(A,C),p1043_2(C,B).
p1043_2(A,B):-move_backwards(A,C),p823_1(C,B).
p1048(A,B):-move_left(A,C),p1048_1(C,B).
p1048_1(A,B):-p1099(A,C),p1048_2(C,B).
p1048_2(A,B):-p450(A,C),p441(C,B).
p1094(A,B):-p1099(A,C),p1094_1(C,B).
p1094_1(A,B):-grab_ball(A,C),p1094_2(C,B).
p1094_2(A,B):-p441(A,C),p823(C,B).
p1103(A,B):-p497(A,C),p1103_1(C,B).
p1103_1(A,B):-p450_1(A,C),p1103_2(C,B).
p1103_2(A,B):-p823(A,C),move_backwards(C,B).
p1104(A,B):-p1197(A,C),p1104_1(C,B).
p1104_1(A,B):-p450_1(A,C),p1104_2(C,B).
p1104_2(A,B):-p260(A,C),p291(C,B).
p1114(A,B):-move_left(A,C),p1114_1(C,B).
p1114_1(A,B):-p450(A,C),p823_1(C,B).
p1117(A,B):-p566(A,C),p1117_1(C,B).
p1117_1(A,B):-grab_ball(A,C),p1117_2(C,B).
p1117_2(A,B):-p149(A,C),p823_1(C,B).
p1124(A,B):-p715(A,C),p1124_1(C,B).
p1124_1(A,B):-grab_ball(A,C),p1124_2(C,B).
p1124_2(A,B):-move_right(A,C),p566_1(C,B).
p1126(A,B):-p823(A,C),p751_1(C,B).
p1127(A,B):-p450(A,C),p1127_1(C,B).
p1127_1(A,B):-move_right(A,C),p1127_2(C,B).
p1127_2(A,B):-drop_ball(A,C),p261(C,B).
p1128(A,B):-move_forwards(A,C),p1128_1(C,B).
p1128_1(A,B):-p566(A,C),p1128_2(C,B).
p1128_2(A,B):-p450_1(A,C),p497(C,B).
p1134(A,B):-p1197(A,C),p1134_1(C,B).
p1134_1(A,B):-p450(A,C),p497(C,B).
p1138(A,B):-p890(A,C),p1138_1(C,B).
p1138_1(A,B):-p450(A,C),p1138_2(C,B).
p1138_2(A,B):-drop_ball(A,C),p566_1(C,B).
p1145(A,B):-move_backwards(A,C),p1145_1(C,B).
p1145_1(A,B):-p450_1(A,C),p1145_2(C,B).
p1145_2(A,B):-drop_ball(A,C),p291(C,B).
p1149(A,B):-p1099(A,C),p1149_1(C,B).
p1149_1(A,B):-p450(A,C),p1149_2(C,B).
p1149_2(A,B):-p823(A,C),p890(C,B).
p1163(A,B):-p450(A,C),p1163_1(C,B).
p1163_1(A,B):-p1068(A,C),p1163_2(C,B).
p1163_2(A,B):-p823(A,C),move_forwards(C,B).
p1167(A,B):-p497(A,C),p1167_1(C,B).
p1167_1(A,B):-grab_ball(A,C),p1167_2(C,B).
p1167_2(A,B):-p823(A,C),p715(C,B).
p1168(A,B):-p566(A,C),p1168_1(C,B).
p1168_1(A,B):-p450_1(A,C),p441(C,B).
p1169(A,B):-move_forwards(A,C),p1169_1(C,B).
p1169_1(A,B):-p450_1(A,C),p1169_2(C,B).
p1169_2(A,B):-p260(A,C),p291(C,B).
p1176(A,B):-p438_1(A,C),p1176_1(C,B).
p1176_1(A,B):-grab_ball(A,C),p751_1(C,B).
p1191(A,B):-p945(A,C),p1191_1(C,B).
p1191_1(A,B):-p450_1(A,C),p1191_2(C,B).
p1191_2(A,B):-p823(A,C),p261(C,B).