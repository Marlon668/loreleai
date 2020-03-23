p164(A,B):-move_left(A,C),move_forwards(C,B).
p197(A,B):-move_backwards(A,C),move_backwards(C,B).
p249(A,B):-move_backwards(A,B).
p286(A,B):-move_left(A,C),move_right(C,B).
p344(A,B):-move_left(A,C),move_forwards(C,B).
p380(A,B):-move_left(A,C),move_right(C,B).
p549(A,B):-move_left(A,B).
p593(A,B):-move_right(A,C),move_backwards(C,B).
p606(A,B):-move_backwards(A,B).
p614(A,B):-move_forwards(A,C),move_forwards(C,B).
p710(A,B):-drop_ball(A,C),move_forwards(C,B).
p797(A,B):-move_forwards(A,B).
p849(A,B):-move_right(A,C),move_right(C,B).
p859(A,B):-move_left(A,B).
p958(A,B):-move_left(A,B).
p989(A,B):-move_left(A,C),move_forwards(C,B).
p1017(A,B):-move_left(A,B).
p1179(A,B):-move_right(A,C),move_backwards(C,B).
p1325(A,B):-move_left(A,C),move_right(C,B).
p38(A,B):-move_left(A,C),p614(C,B).
p43(A,B):-move_forwards(A,C),p43_1(C,B).
p43_1(A,B):-p614(A,C),grab_ball(C,B).
p96(A,B):-move_left(A,C),p96_1(C,B).
p96_1(A,B):-p164(A,C),p614(C,B).
p110(A,B):-move_right(A,C),p110_1(C,B).
p110_1(A,B):-p614(A,C),p614(C,B).
p153(A,B):-p164(A,C),p153_1(C,B).
p153_1(A,B):-p710(A,C),move_right(C,B).
p154(A,B):-move_forwards(A,C),p849(C,B).
p186(A,B):-move_forwards(A,C),p849(C,B).
p193(A,B):-move_right(A,C),p197(C,B).
p195(A,B):-p614(A,C),p614(C,B).
p214(A,B):-move_forwards(A,C),p614(C,B).
p224(A,B):-move_left(A,C),p614(C,B).
p254(A,B):-p197(A,C),p593(C,B).
p262(A,B):-move_right(A,C),p197(C,B).
p327(A,B):-p593(A,C),p327_1(C,B).
p327_1(A,B):-p710(A,C),move_right(C,B).
p349(A,B):-move_right(A,C),p197(C,B).
p406(A,B):-move_right(A,C),p406_1(C,B).
p406_1(A,B):-drop_ball(A,C),p614(C,B).
p410(A,B):-drop_ball(A,C),p410_1(C,B).
p410_1(A,B):-p164(A,C),p614(C,B).
p503(A,B):-p849(A,C),p849(C,B).
p507(A,B):-move_forwards(A,C),p507_1(C,B).
p507_1(A,B):-p849(A,C),p710(C,B).
p552(A,B):-p552_1(A,C),p552_1(C,B).
p552_1(A,B):-move_left(A,C),move_left(C,B).
p647(A,B):-move_right(A,C),p647_1(C,B).
p647_1(A,B):-p614(A,C),p849(C,B).
p700(A,B):-p849(A,C),p700_1(C,B).
p700_1(A,B):-drop_ball(A,C),move_backwards(C,B).
p861(A,B):-p197(A,C),p593(C,B).
p931(A,B):-move_left(A,C),p614(C,B).
p979(A,B):-move_left(A,C),p979_1(C,B).
p979_1(A,B):-p197(A,C),p197(C,B).
p1047(A,B):-move_right(A,C),p614(C,B).
p1116(A,B):-move_right(A,C),p849(C,B).
p1177(A,B):-move_right(A,C),p1177_1(C,B).
p1177_1(A,B):-p197(A,C),p849(C,B).
p1216(A,B):-p593(A,C),p849(C,B).
p1226(A,B):-move_left(A,C),p197(C,B).
p1237(A,B):-move_right(A,C),p614(C,B).
p1260(A,B):-move_forwards(A,C),p1260_1(C,B).
p1260_1(A,B):-drop_ball(A,C),move_right(C,B).
p1272(A,B):-p164(A,C),p164(C,B).
p3(A,B):-p593(A,C),p3_1(C,B).
p3_1(A,B):-p43_1(A,C),p3_2(C,B).
p3_2(A,B):-move_right(A,C),p153(C,B).
p7(A,B):-move_backwards(A,C),p7_1(C,B).
p7_1(A,B):-p327(A,C),move_right(C,B).
p12(A,B):-p43(A,C),p12_1(C,B).
p12_1(A,B):-p979_1(A,C),p12_2(C,B).
p12_2(A,B):-p153(A,C),p193(C,B).
p15(A,B):-move_forwards(A,C),p15_1(C,B).
p15_1(A,B):-p43(A,C),p15_2(C,B).
p15_2(A,B):-p1216(A,C),drop_ball(C,B).
p20(A,B):-p979(A,C),p20_1(C,B).
p20_1(A,B):-p43(A,C),p406(C,B).
p33(A,B):-p43_1(A,C),p33_1(C,B).
p33_1(A,B):-move_forwards(A,C),p33_2(C,B).
p33_2(A,B):-drop_ball(A,C),p1226(C,B).
p39(A,B):-p552_1(A,C),p39_1(C,B).
p39_1(A,B):-grab_ball(A,C),p39_2(C,B).
p39_2(A,B):-p552_1(A,C),p1260(C,B).
p46(A,B):-p593(A,C),p46_1(C,B).
p46_1(A,B):-p1177(A,C),p46_2(C,B).
p46_2(A,B):-grab_ball(A,C),move_left(C,B).
p49(A,B):-grab_ball(A,C),p49_1(C,B).
p49_1(A,B):-p254(A,C),p49_2(C,B).
p49_2(A,B):-drop_ball(A,C),p552_1(C,B).
p50(A,B):-p1216(A,C),p50_1(C,B).
p50_1(A,B):-grab_ball(A,C),p50_2(C,B).
p50_2(A,B):-p153(A,C),p979(C,B).
p56(A,B):-move_right(A,C),p56_1(C,B).
p56_1(A,B):-grab_ball(A,C),p56_2(C,B).
p56_2(A,B):-p1226(A,C),p1260(C,B).
p68(A,B):-p552(A,C),p254(C,B).
p72(A,B):-p43(A,C),p72_1(C,B).
p72_1(A,B):-p979(A,C),p72_2(C,B).
p72_2(A,B):-p153(A,C),p849(C,B).
p83(A,B):-move_backwards(A,C),p83_1(C,B).
p83_1(A,B):-grab_ball(A,C),p83_2(C,B).
p83_2(A,B):-p153(A,C),p647_1(C,B).
p84(A,B):-p593(A,C),p84_1(C,B).
p84_1(A,B):-grab_ball(A,C),p84_2(C,B).
p84_2(A,B):-move_left(A,C),p406_1(C,B).
p92(A,B):-p43_1(A,C),p92_1(C,B).
p92_1(A,B):-p849(A,C),p92_2(C,B).
p92_2(A,B):-drop_ball(A,C),p1226(C,B).
p98(A,B):-p1226(A,C),p410(C,B).
p99(A,B):-p43_1(A,C),p99_1(C,B).
p99_1(A,B):-p197(A,C),p99_2(C,B).
p99_2(A,B):-p327(A,C),p614(C,B).
p108(A,B):-p43(A,C),p108_1(C,B).
p108_1(A,B):-p1226(A,C),p108_2(C,B).
p108_2(A,B):-drop_ball(A,C),p647_1(C,B).
p112(A,B):-move_left(A,C),p112_1(C,B).
p112_1(A,B):-grab_ball(A,C),p112_2(C,B).
p112_2(A,B):-p164(A,C),p507(C,B).
p121(A,B):-move_left(A,C),p121_1(C,B).
p121_1(A,B):-p552_1(A,C),p121_2(C,B).
p121_2(A,B):-grab_ball(A,C),p1177_1(C,B).
p122(A,B):-p1177(A,C),p122_1(C,B).
p122_1(A,B):-grab_ball(A,C),p122_2(C,B).
p122_2(A,B):-p38(A,C),p700_1(C,B).
p129(A,B):-p593(A,C),p129_1(C,B).
p129_1(A,B):-p43_1(A,C),p129_2(C,B).
p129_2(A,B):-p1260(A,C),p552(C,B).
p131(A,B):-p552_1(A,C),p131_1(C,B).
p131_1(A,B):-grab_ball(A,C),p131_2(C,B).
p131_2(A,B):-p1047(A,C),drop_ball(C,B).
p132(A,B):-p43_1(A,C),p132_1(C,B).
p132_1(A,B):-p254(A,C),p132_2(C,B).
p132_2(A,B):-p710(A,C),p552_1(C,B).
p140(A,B):-move_left(A,C),p140_1(C,B).
p140_1(A,B):-p43_1(A,C),p979_1(C,B).
p142(A,B):-grab_ball(A,C),p142_1(C,B).
p142_1(A,B):-p197(A,C),p142_2(C,B).
p142_2(A,B):-p507_1(A,C),p164(C,B).
p145(A,B):-move_right(A,C),p145_1(C,B).
p145_1(A,B):-grab_ball(A,C),p145_2(C,B).
p145_2(A,B):-p1260(A,C),p552_1(C,B).
p148(A,B):-p43(A,C),p148_1(C,B).
p148_1(A,B):-p593(A,C),p148_2(C,B).
p148_2(A,B):-p1177(A,C),p410(C,B).
p155(A,B):-p254(A,C),p155_1(C,B).
p155_1(A,B):-grab_ball(A,C),p155_2(C,B).
p155_2(A,B):-p406(A,C),p1047(C,B).
p172(A,B):-p193(A,C),p552(C,B).
p174(A,B):-move_right(A,C),p174_1(C,B).
p174_1(A,B):-p43(A,C),p174_2(C,B).
p174_2(A,B):-move_backwards(A,C),p507_1(C,B).
p177(A,B):-p1216(A,C),p177_1(C,B).
p177_1(A,B):-grab_ball(A,C),p177_2(C,B).
p177_2(A,B):-p552_1(A,C),p1226(C,B).
p185(A,B):-p164(A,C),p185_1(C,B).
p185_1(A,B):-p552_1(A,C),p185_2(C,B).
p185_2(A,B):-p153(A,C),p979_1(C,B).
p194(A,B):-p164(A,C),p194_1(C,B).
p194_1(A,B):-p43(A,C),p194_2(C,B).
p194_2(A,B):-p1177(A,C),p1260_1(C,B).
p203(A,B):-move_right(A,C),p203_1(C,B).
p203_1(A,B):-p43_1(A,C),p203_2(C,B).
p203_2(A,B):-p1226(A,C),p710(C,B).
p205(A,B):-p154(A,C),p205_1(C,B).
p205_1(A,B):-drop_ball(A,C),p1226(C,B).
p206(A,B):-p647(A,C),p206_1(C,B).
p206_1(A,B):-grab_ball(A,C),p206_2(C,B).
p206_2(A,B):-p1272(A,C),p700_1(C,B).
p215(A,B):-move_backwards(A,C),p215_1(C,B).
p215_1(A,B):-grab_ball(A,C),p215_2(C,B).
p215_2(A,B):-p406(A,C),p1226(C,B).
p219(A,B):-move_right(A,C),p219_1(C,B).
p219_1(A,B):-p43_1(A,C),p219_2(C,B).
p219_2(A,B):-move_left(A,C),p1260(C,B).
p243(A,B):-p43_1(A,C),p243_1(C,B).
p243_1(A,B):-move_left(A,C),p243_2(C,B).
p243_2(A,B):-p1226(A,C),p153_1(C,B).
p244(A,B):-p1116(A,C),p244_1(C,B).
p244_1(A,B):-p43(A,C),p244_2(C,B).
p244_2(A,B):-p193(A,C),p700_1(C,B).
p257(A,B):-move_backwards(A,C),p257_1(C,B).
p257_1(A,B):-grab_ball(A,C),p327(C,B).
p258(A,B):-p1177_1(A,C),p258_1(C,B).
p258_1(A,B):-grab_ball(A,C),p700(C,B).
p263(A,B):-p43_1(A,C),p263_1(C,B).
p263_1(A,B):-move_left(A,C),p263_2(C,B).
p263_2(A,B):-drop_ball(A,C),p1177_1(C,B).
p266(A,B):-p1116(A,C),p266_1(C,B).
p266_1(A,B):-grab_ball(A,C),p266_2(C,B).
p266_2(A,B):-move_forwards(A,C),p700_1(C,B).
p267(A,B):-p43_1(A,C),p267_1(C,B).
p267_1(A,B):-p1177_1(A,C),p267_2(C,B).
p267_2(A,B):-p327(A,C),p214(C,B).
p271(A,B):-p193(A,C),p271_1(C,B).
p271_1(A,B):-grab_ball(A,C),move_forwards(C,B).
p272(A,B):-p552_1(A,C),p272_1(C,B).
p272_1(A,B):-grab_ball(A,C),p272_2(C,B).
p272_2(A,B):-p154(A,C),p153_1(C,B).
p275(A,B):-p1216(A,C),p275_1(C,B).
p275_1(A,B):-grab_ball(A,C),p275_2(C,B).
p275_2(A,B):-p1260(A,C),p979(C,B).
p278(A,B):-p979(A,C),p278_1(C,B).
p278_1(A,B):-p43_1(A,C),p278_2(C,B).
p278_2(A,B):-move_left(A,C),p1260(C,B).
p279(A,B):-p43_1(A,C),p279_1(C,B).
p279_1(A,B):-p164(A,C),p1260(C,B).
p280(A,B):-p593(A,C),p280_1(C,B).
p280_1(A,B):-p43_1(A,C),p280_2(C,B).
p280_2(A,B):-p593(A,C),p327(C,B).
p285(A,B):-move_right(A,C),p285_1(C,B).
p285_1(A,B):-grab_ball(A,C),p285_2(C,B).
p285_2(A,B):-p614(A,C),p153(C,B).
p290(A,B):-p593(A,C),p290_1(C,B).
p290_1(A,B):-grab_ball(A,C),p290_2(C,B).
p290_2(A,B):-p614(A,C),drop_ball(C,B).
p304(A,B):-p43(A,C),p304_1(C,B).
p304_1(A,B):-move_backwards(A,C),p304_2(C,B).
p304_2(A,B):-p700(A,C),p164(C,B).
p313(A,B):-p552_1(A,C),p313_1(C,B).
p313_1(A,B):-p1260(A,C),p1226(C,B).
p322(A,B):-p552_1(A,C),p322_1(C,B).
p322_1(A,B):-grab_ball(A,C),p322_2(C,B).
p322_2(A,B):-move_right(A,C),p1260_1(C,B).
p326(A,B):-p197(A,C),p326_1(C,B).
p326_1(A,B):-grab_ball(A,C),p326_2(C,B).
p326_2(A,B):-p406(A,C),p552_1(C,B).
p330(A,B):-p43(A,C),p330_1(C,B).
p330_1(A,B):-p979_1(A,C),p330_2(C,B).
p330_2(A,B):-p1260(A,C),p110(C,B).
p332(A,B):-move_right(A,C),p332_1(C,B).
p332_1(A,B):-p43(A,C),p332_2(C,B).
p332_2(A,B):-move_right(A,C),p1260(C,B).
p338(A,B):-move_forwards(A,C),p338_1(C,B).
p338_1(A,B):-grab_ball(A,C),p338_2(C,B).
p338_2(A,B):-p593(A,C),p700_1(C,B).
p348(A,B):-p849(A,C),p348_1(C,B).
p348_1(A,B):-p43(A,C),p348_2(C,B).
p348_2(A,B):-p593(A,C),p700_1(C,B).
p357(A,B):-grab_ball(A,C),p357_1(C,B).
p357_1(A,B):-p96(A,C),p700(C,B).
p363(A,B):-grab_ball(A,C),p363_1(C,B).
p363_1(A,B):-p1226(A,C),p363_2(C,B).
p363_2(A,B):-p1260(A,C),p197(C,B).
p372(A,B):-p979(A,C),p372_1(C,B).
p372_1(A,B):-p43(A,C),p372_2(C,B).
p372_2(A,B):-p154(A,C),p700(C,B).
p374(A,B):-p43_1(A,C),p374_1(C,B).
p374_1(A,B):-move_backwards(A,C),p700(C,B).
p391(A,B):-p43_1(A,C),p391_1(C,B).
p391_1(A,B):-p1226(A,C),p1260_1(C,B).
p399(A,B):-p110_1(A,C),p399_1(C,B).
p399_1(A,B):-p700(A,C),p399_2(C,B).
p399_2(A,B):-move_backwards(A,C),p552(C,B).
p404(A,B):-p979(A,C),p404_1(C,B).
p404_1(A,B):-grab_ball(A,C),p404_2(C,B).
p404_2(A,B):-p700(A,C),p96(C,B).
p405(A,B):-move_left(A,C),p405_1(C,B).
p405_1(A,B):-p43(A,C),p405_2(C,B).
p405_2(A,B):-p197(A,C),p1260_1(C,B).
p412(A,B):-p979(A,C),p412_1(C,B).
p412_1(A,B):-p43_1(A,C),p412_2(C,B).
p412_2(A,B):-p38(A,C),p710(C,B).
p430(A,B):-move_left(A,C),p430_1(C,B).
p430_1(A,B):-p43(A,C),p700(C,B).
p431(A,B):-move_forwards(A,C),p431_1(C,B).
p431_1(A,B):-p43(A,C),p431_2(C,B).
p431_2(A,B):-p254(A,C),p710(C,B).
p432(A,B):-p193(A,C),p432_1(C,B).
p432_1(A,B):-grab_ball(A,C),p432_2(C,B).
p432_2(A,B):-p1216(A,C),p406_1(C,B).
p438(A,B):-p552(A,C),p438_1(C,B).
p438_1(A,B):-p43_1(A,C),p438_2(C,B).
p438_2(A,B):-p507_1(A,C),p154(C,B).
p442(A,B):-move_left(A,C),p442_1(C,B).
p442_1(A,B):-p43(A,C),move_right(C,B).
p443(A,B):-p593(A,C),p443_1(C,B).
p443_1(A,B):-p43_1(A,C),p443_2(C,B).
p443_2(A,B):-p1216(A,C),p410(C,B).
p466(A,B):-p593(A,C),p552(C,B).
p469(A,B):-move_right(A,C),p469_1(C,B).
p469_1(A,B):-p43_1(A,C),p469_2(C,B).
p469_2(A,B):-move_left(A,C),p327(C,B).
p470(A,B):-move_forwards(A,C),p470_1(C,B).
p470_1(A,B):-grab_ball(A,C),p470_2(C,B).
p470_2(A,B):-p552_1(A,C),p700_1(C,B).
p483(A,B):-p254(A,C),p552(C,B).
p490(A,B):-move_left(A,C),p490_1(C,B).
p490_1(A,B):-p43_1(A,C),p490_2(C,B).
p490_2(A,B):-p849(A,C),p1260(C,B).
p506(A,B):-p43(A,C),p506_1(C,B).
p506_1(A,B):-p979_1(A,C),p506_2(C,B).
p506_2(A,B):-p1260(A,C),move_backwards(C,B).
p518(A,B):-move_right(A,C),p518_1(C,B).
p518_1(A,B):-grab_ball(A,C),p518_2(C,B).
p518_2(A,B):-p96_1(A,C),p700(C,B).
p525(A,B):-p254(A,C),p525_1(C,B).
p525_1(A,B):-grab_ball(A,C),p525_2(C,B).
p525_2(A,B):-p507_1(A,C),p1272(C,B).
p526(A,B):-p552_1(A,C),p526_1(C,B).
p526_1(A,B):-p979(A,C),p526_2(C,B).
p526_2(A,B):-grab_ball(A,C),p503(C,B).
p527(A,B):-p164(A,C),p527_1(C,B).
p527_1(A,B):-p979(A,C),p527_2(C,B).
p527_2(A,B):-grab_ball(A,C),p647(C,B).
p530(A,B):-p164(A,C),p530_1(C,B).
p530_1(A,B):-grab_ball(A,C),p327(C,B).
p533(A,B):-p43_1(A,C),p533_1(C,B).
p533_1(A,B):-move_left(A,C),p533_2(C,B).
p533_2(A,B):-p700_1(A,C),p552_1(C,B).
p536(A,B):-grab_ball(A,C),p536_1(C,B).
p536_1(A,B):-p1226(A,C),p536_2(C,B).
p536_2(A,B):-p700_1(A,C),p552_1(C,B).
p539(A,B):-p197(A,C),p539_1(C,B).
p539_1(A,B):-grab_ball(A,C),p539_2(C,B).
p539_2(A,B):-p507(A,C),p647_1(C,B).
p559(A,B):-p979(A,C),p559_1(C,B).
p559_1(A,B):-p43_1(A,C),p559_2(C,B).
p559_2(A,B):-p406(A,C),p979(C,B).
p570(A,B):-p552_1(A,C),p570_1(C,B).
p570_1(A,B):-grab_ball(A,C),p570_2(C,B).
p570_2(A,B):-p406(A,C),p647_1(C,B).
p572(A,B):-move_left(A,C),p572_1(C,B).
p572_1(A,B):-grab_ball(A,C),p572_2(C,B).
p572_2(A,B):-p197(A,C),p700_1(C,B).
p579(A,B):-p1177(A,C),p579_1(C,B).
p579_1(A,B):-grab_ball(A,C),p579_2(C,B).
p579_2(A,B):-p164(A,C),p406_1(C,B).
p580(A,B):-p43(A,C),p580_1(C,B).
p580_1(A,B):-p197(A,C),p580_2(C,B).
p580_2(A,B):-drop_ball(A,C),p1116(C,B).
p581(A,B):-p43_1(A,C),p581_1(C,B).
p581_1(A,B):-p193(A,C),drop_ball(C,B).
p582(A,B):-move_backwards(A,C),p582_1(C,B).
p582_1(A,B):-grab_ball(A,C),p582_2(C,B).
p582_2(A,B):-p110(A,C),p327(C,B).
p596(A,B):-p503(A,C),p596_1(C,B).
p596_1(A,B):-grab_ball(A,C),p596_2(C,B).
p596_2(A,B):-move_backwards(A,C),p1226(C,B).
p600(A,B):-move_left(A,C),p600_1(C,B).
p600_1(A,B):-grab_ball(A,C),p600_2(C,B).
p600_2(A,B):-p849(A,C),p507_1(C,B).
p602(A,B):-p154(A,C),p602_1(C,B).
p602_1(A,B):-grab_ball(A,C),p602_2(C,B).
p602_2(A,B):-p1260(A,C),move_backwards(C,B).
p607(A,B):-grab_ball(A,C),p607_1(C,B).
p607_1(A,B):-move_left(A,C),p406_1(C,B).
p615(A,B):-move_right(A,C),p615_1(C,B).
p615_1(A,B):-grab_ball(A,C),p615_2(C,B).
p615_2(A,B):-p1116(A,C),p410(C,B).
p622(A,B):-p979(A,C),p622_1(C,B).
p622_1(A,B):-p43_1(A,C),p193(C,B).
p625(A,B):-move_forwards(A,C),p625_1(C,B).
p625_1(A,B):-grab_ball(A,C),p625_2(C,B).
p625_2(A,B):-p1260(A,C),p979_1(C,B).
p630(A,B):-p43_1(A,C),p630_1(C,B).
p630_1(A,B):-p197(A,C),p630_2(C,B).
p630_2(A,B):-drop_ball(A,C),p214(C,B).
p645(A,B):-p1177(A,C),p645_1(C,B).
p645_1(A,B):-grab_ball(A,C),p406(C,B).
p649(A,B):-move_forwards(A,C),p649_1(C,B).
p649_1(A,B):-p43(A,C),p649_2(C,B).
p649_2(A,B):-p197(A,C),p1260_1(C,B).
p652(A,B):-p197(A,C),p652_1(C,B).
p652_1(A,B):-p552(A,C),p652_2(C,B).
p652_2(A,B):-drop_ball(A,C),p214(C,B).
p659(A,B):-move_right(A,C),p659_1(C,B).
p659_1(A,B):-grab_ball(A,C),p659_2(C,B).
p659_2(A,B):-p1260(A,C),p254(C,B).
p663(A,B):-p593(A,C),p663_1(C,B).
p663_1(A,B):-grab_ball(A,C),p663_2(C,B).
p663_2(A,B):-p406(A,C),p1226(C,B).
p665(A,B):-p849(A,C),p665_1(C,B).
p665_1(A,B):-grab_ball(A,C),p665_2(C,B).
p665_2(A,B):-p153(A,C),p593(C,B).
p668(A,B):-p43_1(A,C),p668_1(C,B).
p668_1(A,B):-p552_1(A,C),p710(C,B).
p669(A,B):-p979(A,C),p669_1(C,B).
p669_1(A,B):-p43(A,C),p507_1(C,B).
p673(A,B):-move_right(A,C),p673_1(C,B).
p673_1(A,B):-grab_ball(A,C),p673_2(C,B).
p673_2(A,B):-p154(A,C),p406(C,B).
p675(A,B):-p43(A,C),p675_1(C,B).
p675_1(A,B):-move_left(A,C),p675_2(C,B).
p675_2(A,B):-drop_ball(A,C),p193(C,B).
p676(A,B):-p254(A,C),p676_1(C,B).
p676_1(A,B):-grab_ball(A,C),p676_2(C,B).
p676_2(A,B):-p507_1(A,C),p1272(C,B).
p681(A,B):-move_left(A,C),p681_1(C,B).
p681_1(A,B):-p43(A,C),p700(C,B).
p687(A,B):-p593(A,C),p687_1(C,B).
p687_1(A,B):-p507_1(A,C),p687_2(C,B).
p687_2(A,B):-p552(A,C),p593(C,B).
p693(A,B):-p849(A,C),p693_1(C,B).
p693_1(A,B):-grab_ball(A,C),p693_2(C,B).
p693_2(A,B):-move_left(A,C),p164(C,B).
p696(A,B):-p43_1(A,C),p696_1(C,B).
p696_1(A,B):-p153(A,C),move_right(C,B).
p697(A,B):-grab_ball(A,C),p697_1(C,B).
p697_1(A,B):-p164(A,C),p697_2(C,B).
p697_2(A,B):-p700_1(A,C),p1177(C,B).
p702(A,B):-grab_ball(A,C),p702_1(C,B).
p702_1(A,B):-move_left(A,C),p702_2(C,B).
p702_2(A,B):-p254(A,C),p700_1(C,B).
p712(A,B):-p43_1(A,C),p712_1(C,B).
p712_1(A,B):-move_left(A,C),p712_2(C,B).
p712_2(A,B):-p1260(A,C),p552_1(C,B).
p716(A,B):-p849(A,C),p716_1(C,B).
p716_1(A,B):-p43_1(A,C),p716_2(C,B).
p716_2(A,B):-p1260(A,C),p552_1(C,B).
p717(A,B):-p254(A,C),p717_1(C,B).
p717_1(A,B):-grab_ball(A,C),p717_2(C,B).
p717_2(A,B):-p700(A,C),p164(C,B).
p718(A,B):-p43_1(A,C),p718_1(C,B).
p718_1(A,B):-p979_1(A,C),p1260(C,B).
p727(A,B):-p164(A,C),p727_1(C,B).
p727_1(A,B):-p43(A,C),p727_2(C,B).
p727_2(A,B):-p197(A,C),p507_1(C,B).
p732(A,B):-p43_1(A,C),p732_1(C,B).
p732_1(A,B):-p1260(A,C),p1177_1(C,B).
p737(A,B):-p43(A,C),p737_1(C,B).
p737_1(A,B):-move_left(A,C),p737_2(C,B).
p737_2(A,B):-p507_1(A,C),p1226(C,B).
p739(A,B):-move_left(A,C),p739_1(C,B).
p739_1(A,B):-p979(A,C),p739_2(C,B).
p739_2(A,B):-p43_1(A,C),p214(C,B).
p743(A,B):-move_backwards(A,C),p743_1(C,B).
p743_1(A,B):-grab_ball(A,C),p743_2(C,B).
p743_2(A,B):-p327(A,C),p614(C,B).
p747(A,B):-p1177(A,C),p747_1(C,B).
p747_1(A,B):-grab_ball(A,C),p747_2(C,B).
p747_2(A,B):-p1260(A,C),move_backwards(C,B).
p751(A,B):-move_left(A,C),p751_1(C,B).
p751_1(A,B):-p43_1(A,C),p751_2(C,B).
p751_2(A,B):-move_left(A,C),p700_1(C,B).
p752(A,B):-p552_1(A,C),p752_1(C,B).
p752_1(A,B):-grab_ball(A,C),p752_2(C,B).
p752_2(A,B):-p1260(A,C),p614(C,B).
p756(A,B):-p43_1(A,C),p756_1(C,B).
p756_1(A,B):-p1226(A,C),p756_2(C,B).
p756_2(A,B):-drop_ball(A,C),move_left(C,B).
p757(A,B):-p43_1(A,C),p757_1(C,B).
p757_1(A,B):-p164(A,C),p700_1(C,B).
p768(A,B):-p43_1(A,C),p1177(C,B).
p776(A,B):-move_left(A,C),p776_1(C,B).
p776_1(A,B):-p507(A,C),p1216(C,B).
p794(A,B):-p979(A,C),p794_1(C,B).
p794_1(A,B):-p43_1(A,C),p794_2(C,B).
p794_2(A,B):-move_forwards(A,C),p406_1(C,B).
p795(A,B):-p164(A,C),p795_1(C,B).
p795_1(A,B):-grab_ball(A,C),p795_2(C,B).
p795_2(A,B):-p1260(A,C),p1226(C,B).
p796(A,B):-p164(A,C),p796_1(C,B).
p796_1(A,B):-grab_ball(A,C),p796_2(C,B).
p796_2(A,B):-move_left(A,C),p700(C,B).
p800(A,B):-p979(A,C),p800_1(C,B).
p800_1(A,B):-p43(A,C),p800_2(C,B).
p800_2(A,B):-p507(A,C),p552_1(C,B).
p812(A,B):-p43(A,C),p1177(C,B).
p831(A,B):-move_left(A,C),p831_1(C,B).
p831_1(A,B):-grab_ball(A,C),p831_2(C,B).
p831_2(A,B):-p979_1(A,C),p507(C,B).
p844(A,B):-p96(A,C),p844_1(C,B).
p844_1(A,B):-grab_ball(A,C),p844_2(C,B).
p844_2(A,B):-p700(A,C),p552(C,B).
p845(A,B):-p593(A,C),p845_1(C,B).
p845_1(A,B):-grab_ball(A,C),p845_2(C,B).
p845_2(A,B):-p507_1(A,C),p193(C,B).
p851(A,B):-move_left(A,C),p851_1(C,B).
p851_1(A,B):-p552_1(A,C),p851_2(C,B).
p851_2(A,B):-grab_ball(A,C),p406(C,B).
p853(A,B):-p1226(A,C),p1260_1(C,B).
p866(A,B):-p593(A,C),p866_1(C,B).
p866_1(A,B):-p43_1(A,C),p866_2(C,B).
p866_2(A,B):-move_left(A,C),p327(C,B).
p867(A,B):-p1116(A,C),p867_1(C,B).
p867_1(A,B):-grab_ball(A,C),p552(C,B).
p880(A,B):-p164(A,C),p880_1(C,B).
p880_1(A,B):-grab_ball(A,C),p880_2(C,B).
p880_2(A,B):-p1177_1(A,C),p1260_1(C,B).
p883(A,B):-p96(A,C),p552_1(C,B).
p885(A,B):-grab_ball(A,C),p885_1(C,B).
p885_1(A,B):-p164(A,C),p885_2(C,B).
p885_2(A,B):-p979(A,C),p406_1(C,B).
p888(A,B):-p552_1(A,C),p888_1(C,B).
p888_1(A,B):-grab_ball(A,C),p888_2(C,B).
p888_2(A,B):-p614(A,C),p406_1(C,B).
p891(A,B):-p1226(A,C),p891_1(C,B).
p891_1(A,B):-p153(A,C),p614(C,B).
p892(A,B):-p110(A,C),p892_1(C,B).
p892_1(A,B):-grab_ball(A,C),p892_2(C,B).
p892_2(A,B):-move_backwards(A,C),p552(C,B).
p900(A,B):-p193(A,C),p900_1(C,B).
p900_1(A,B):-grab_ball(A,C),p900_2(C,B).
p900_2(A,B):-p164(A,C),p700_1(C,B).
p908(A,B):-p164(A,C),p908_1(C,B).
p908_1(A,B):-grab_ball(A,C),p908_2(C,B).
p908_2(A,B):-p507_1(A,C),p164(C,B).
p917(A,B):-move_left(A,C),p917_1(C,B).
p917_1(A,B):-grab_ball(A,C),p917_2(C,B).
p917_2(A,B):-p38(A,C),p700(C,B).
p920(A,B):-p43(A,C),p920_1(C,B).
p920_1(A,B):-p593(A,C),p920_2(C,B).
p920_2(A,B):-drop_ball(A,C),p552_1(C,B).
p932(A,B):-grab_ball(A,C),p932_1(C,B).
p932_1(A,B):-p154(A,C),p932_2(C,B).
p932_2(A,B):-p1260(A,C),p193(C,B).
p935(A,B):-p193(A,C),p935_1(C,B).
p935_1(A,B):-grab_ball(A,C),p935_2(C,B).
p935_2(A,B):-move_forwards(A,C),p700(C,B).
p939(A,B):-p849(A,C),p939_1(C,B).
p939_1(A,B):-p43_1(A,C),p939_2(C,B).
p939_2(A,B):-p979_1(A,C),p406_1(C,B).
p953(A,B):-move_right(A,C),p953_1(C,B).
p953_1(A,B):-p43_1(A,C),p953_2(C,B).
p953_2(A,B):-move_left(A,C),p552_1(C,B).
p954(A,B):-p43(A,C),p954_1(C,B).
p954_1(A,B):-p164(A,C),p700_1(C,B).
p955(A,B):-move_left(A,C),p955_1(C,B).
p955_1(A,B):-p979(A,C),p955_2(C,B).
p955_2(A,B):-p153(A,C),p1116(C,B).
p961(A,B):-grab_ball(A,C),p961_1(C,B).
p961_1(A,B):-p552_1(A,C),p961_2(C,B).
p961_2(A,B):-p1260(A,C),p1226(C,B).
p964(A,B):-p1116(A,C),p964_1(C,B).
p964_1(A,B):-p43(A,C),p964_2(C,B).
p964_2(A,B):-p979(A,C),p164(C,B).
p972(A,B):-p197(A,C),p972_1(C,B).
p972_1(A,B):-grab_ball(A,C),p972_2(C,B).
p972_2(A,B):-p1260(A,C),p614(C,B).
p985(A,B):-p43(A,C),p985_1(C,B).
p985_1(A,B):-move_left(A,C),p985_2(C,B).
p985_2(A,B):-p1216(A,C),drop_ball(C,B).
p988(A,B):-p164(A,C),p988_1(C,B).
p988_1(A,B):-grab_ball(A,C),p988_2(C,B).
p988_2(A,B):-p979_1(A,C),p1260(C,B).
p999(A,B):-move_left(A,C),p999_1(C,B).
p999_1(A,B):-grab_ball(A,C),p999_2(C,B).
p999_2(A,B):-p1216(A,C),p406(C,B).
p1000(A,B):-p43_1(A,C),p1000_1(C,B).
p1000_1(A,B):-move_left(A,C),p1000_2(C,B).
p1000_2(A,B):-drop_ball(A,C),p164(C,B).
p1001(A,B):-p979(A,C),p1001_1(C,B).
p1001_1(A,B):-p43(A,C),p1001_2(C,B).
p1001_2(A,B):-p254(A,C),drop_ball(C,B).
p1004(A,B):-move_left(A,C),p1004_1(C,B).
p1004_1(A,B):-p552_1(A,C),p1004_2(C,B).
p1004_2(A,B):-p43_1(A,C),p507_1(C,B).
p1012(A,B):-p1216(A,C),p1012_1(C,B).
p1012_1(A,B):-grab_ball(A,C),p1012_2(C,B).
p1012_2(A,B):-p164(A,C),p153(C,B).
p1013(A,B):-p254(A,C),p1013_1(C,B).
p1013_1(A,B):-grab_ball(A,C),p1013_2(C,B).
p1013_2(A,B):-p507(A,C),p614(C,B).
p1015(A,B):-p154(A,C),p1015_1(C,B).
p1015_1(A,B):-p1260(A,C),p193(C,B).
p1023(A,B):-p43(A,C),p1023_1(C,B).
p1023_1(A,B):-p197(A,C),p1023_2(C,B).
p1023_2(A,B):-p406(A,C),p1226(C,B).
p1030(A,B):-move_left(A,C),p1030_1(C,B).
p1030_1(A,B):-grab_ball(A,C),p1030_2(C,B).
p1030_2(A,B):-p1260(A,C),p552_1(C,B).
p1033(A,B):-p43(A,C),p1033_1(C,B).
p1033_1(A,B):-p979_1(A,C),p1033_2(C,B).
p1033_2(A,B):-p153(A,C),p614(C,B).
p1034(A,B):-move_right(A,C),p1034_1(C,B).
p1034_1(A,B):-p43_1(A,C),p1034_2(C,B).
p1034_2(A,B):-p254(A,C),p406_1(C,B).
p1039(A,B):-p647(A,C),p1039_1(C,B).
p1039_1(A,B):-grab_ball(A,C),p1039_2(C,B).
p1039_2(A,B):-p406(A,C),p979(C,B).
p1042(A,B):-p43_1(A,C),p1042_1(C,B).
p1042_1(A,B):-move_forwards(A,C),p1042_2(C,B).
p1042_2(A,B):-p700(A,C),p164(C,B).
p1048(A,B):-move_right(A,C),p1048_1(C,B).
p1048_1(A,B):-grab_ball(A,C),p1048_2(C,B).
p1048_2(A,B):-p164(A,C),drop_ball(C,B).
p1056(A,B):-p647(A,C),p1056_1(C,B).
p1056_1(A,B):-grab_ball(A,C),p1056_2(C,B).
p1056_2(A,B):-move_left(A,C),p153_1(C,B).
p1057(A,B):-p1116(A,C),p1057_1(C,B).
p1057_1(A,B):-p43(A,C),p1057_2(C,B).
p1057_2(A,B):-p1226(A,C),p1260(C,B).
p1063(A,B):-p593(A,C),p1063_1(C,B).
p1063_1(A,B):-p43_1(A,C),p1063_2(C,B).
p1063_2(A,B):-p197(A,C),p153(C,B).
p1072(A,B):-p154(A,C),p1072_1(C,B).
p1072_1(A,B):-p43(A,C),p1072_2(C,B).
p1072_2(A,B):-p552(A,C),p1260_1(C,B).
p1073(A,B):-move_left(A,C),p1073_1(C,B).
p1073_1(A,B):-grab_ball(A,C),p1073_2(C,B).
p1073_2(A,B):-p614(A,C),drop_ball(C,B).
p1078(A,B):-p154(A,C),p1078_1(C,B).
p1078_1(A,B):-p43(A,C),p1078_2(C,B).
p1078_2(A,B):-p552(A,C),p254(C,B).
p1080(A,B):-p552(A,C),p1080_1(C,B).
p1080_1(A,B):-grab_ball(A,C),p1080_2(C,B).
p1080_2(A,B):-p647_1(A,C),p700(C,B).
p1083(A,B):-grab_ball(A,C),p1083_1(C,B).
p1083_1(A,B):-p1260(A,C),p1083_2(C,B).
p1083_2(A,B):-p197(A,C),p552(C,B).
p1085(A,B):-grab_ball(A,C),p1085_1(C,B).
p1085_1(A,B):-p193(A,C),p1085_2(C,B).
p1085_2(A,B):-p700_1(A,C),p552(C,B).
p1088(A,B):-p43_1(A,C),p1088_1(C,B).
p1088_1(A,B):-p979_1(A,C),p710(C,B).
p1094(A,B):-p1216(A,C),p1094_1(C,B).
p1094_1(A,B):-grab_ball(A,C),p1094_2(C,B).
p1094_2(A,B):-p197(A,C),p1260_1(C,B).
p1098(A,B):-grab_ball(A,C),p1098_1(C,B).
p1098_1(A,B):-move_left(A,C),p1098_2(C,B).
p1098_2(A,B):-p327(A,C),p197(C,B).
p1100(A,B):-p979(A,C),p1100_1(C,B).
p1100_1(A,B):-p43(A,C),p503(C,B).
p1101(A,B):-move_left(A,C),p1101_1(C,B).
p1101_1(A,B):-grab_ball(A,C),p1101_2(C,B).
p1101_2(A,B):-p327(A,C),p849(C,B).
p1102(A,B):-p154(A,C),p1102_1(C,B).
p1102_1(A,B):-grab_ball(A,C),p153(C,B).
p1108(A,B):-p43_1(A,C),p1108_1(C,B).
p1108_1(A,B):-p1226(A,C),p1108_2(C,B).
p1108_2(A,B):-p153_1(A,C),p552(C,B).
p1113(A,B):-p43(A,C),p1113_1(C,B).
p1113_1(A,B):-p254(A,C),p1113_2(C,B).
p1113_2(A,B):-p710(A,C),p552_1(C,B).
p1117(A,B):-move_forwards(A,C),p1117_1(C,B).
p1117_1(A,B):-grab_ball(A,C),p1117_2(C,B).
p1117_2(A,B):-p593(A,C),p710(C,B).
p1120(A,B):-move_left(A,C),p1120_1(C,B).
p1120_1(A,B):-move_backwards(A,C),p1120_2(C,B).
p1120_2(A,B):-p410(A,C),p552_1(C,B).
p1121(A,B):-move_left(A,C),p1121_1(C,B).
p1121_1(A,B):-grab_ball(A,C),p1121_2(C,B).
p1121_2(A,B):-p552_1(A,C),p700_1(C,B).
p1129(A,B):-p197(A,C),p1129_1(C,B).
p1129_1(A,B):-grab_ball(A,C),p1129_2(C,B).
p1129_2(A,B):-p164(A,C),p700(C,B).
p1133(A,B):-p849(A,C),p1133_1(C,B).
p1133_1(A,B):-p43(A,C),p1226(C,B).
p1146(A,B):-p1116(A,C),p410(C,B).
p1148(A,B):-p164(A,C),p1148_1(C,B).
p1148_1(A,B):-grab_ball(A,C),p1148_2(C,B).
p1148_2(A,B):-p327(A,C),p214(C,B).
p1159(A,B):-p552(A,C),p1159_1(C,B).
p1159_1(A,B):-p43_1(A,C),p1159_2(C,B).
p1159_2(A,B):-p1260(A,C),p254(C,B).
p1162(A,B):-p1216(A,C),p1162_1(C,B).
p1162_1(A,B):-grab_ball(A,C),p1162_2(C,B).
p1162_2(A,B):-p552(A,C),p406_1(C,B).
p1165(A,B):-p849(A,C),p1165_1(C,B).
p1165_1(A,B):-grab_ball(A,C),p1165_2(C,B).
p1165_2(A,B):-p96_1(A,C),p1260(C,B).
p1166(A,B):-p979(A,C),p1166_1(C,B).
p1166_1(A,B):-grab_ball(A,C),p1166_2(C,B).
p1166_2(A,B):-p1260(A,C),p647_1(C,B).
p1172(A,B):-p979(A,C),p1172_1(C,B).
p1172_1(A,B):-p43(A,C),p1172_2(C,B).
p1172_2(A,B):-p197(A,C),p406(C,B).
p1174(A,B):-p43_1(A,C),p1174_1(C,B).
p1174_1(A,B):-move_left(A,C),p1174_2(C,B).
p1174_2(A,B):-p254(A,C),p406_1(C,B).
p1176(A,B):-p552_1(A,C),p1176_1(C,B).
p1176_1(A,B):-p43_1(A,C),p1176_2(C,B).
p1176_2(A,B):-p593(A,C),drop_ball(C,B).
p1178(A,B):-move_forwards(A,C),p1178_1(C,B).
p1178_1(A,B):-grab_ball(A,C),p1178_2(C,B).
p1178_2(A,B):-p593(A,C),p410(C,B).
p1181(A,B):-p197(A,C),p1181_1(C,B).
p1181_1(A,B):-p552(A,C),p1181_2(C,B).
p1181_2(A,B):-p406(A,C),p849(C,B).
p1187(A,B):-p164(A,C),p1187_1(C,B).
p1187_1(A,B):-grab_ball(A,C),p1187_2(C,B).
p1187_2(A,B):-p979(A,C),p153_1(C,B).
p1194(A,B):-p849(A,C),p1194_1(C,B).
p1194_1(A,B):-grab_ball(A,C),p1194_2(C,B).
p1194_2(A,B):-move_forwards(A,C),p153(C,B).
p1199(A,B):-move_backwards(A,C),p1199_1(C,B).
p1199_1(A,B):-p410(A,C),p552_1(C,B).
p1206(A,B):-move_left(A,C),p1206_1(C,B).
p1206_1(A,B):-p1216(A,C),p1206_2(C,B).
p1206_2(A,B):-grab_ball(A,C),p153(C,B).
p1218(A,B):-p193(A,C),p1218_1(C,B).
p1218_1(A,B):-grab_ball(A,C),p1218_2(C,B).
p1218_2(A,B):-move_forwards(A,C),drop_ball(C,B).
p1220(A,B):-p197(A,C),p1220_1(C,B).
p1220_1(A,B):-grab_ball(A,C),p1220_2(C,B).
p1220_2(A,B):-p38(A,C),p710(C,B).
p1229(A,B):-p197(A,C),p1229_1(C,B).
p1229_1(A,B):-grab_ball(A,C),p1229_2(C,B).
p1229_2(A,B):-p614(A,C),p700_1(C,B).
p1232(A,B):-p43(A,C),p1232_1(C,B).
p1232_1(A,B):-p1116(A,C),p1260(C,B).
p1233(A,B):-p197(A,C),p1233_1(C,B).
p1233_1(A,B):-grab_ball(A,C),p1233_2(C,B).
p1233_2(A,B):-p647(A,C),p700_1(C,B).
p1235(A,B):-p197(A,C),p1235_1(C,B).
p1235_1(A,B):-grab_ball(A,C),p1235_2(C,B).
p1235_2(A,B):-move_left(A,C),p507_1(C,B).
p1238(A,B):-p197(A,C),p1238_1(C,B).
p1238_1(A,B):-grab_ball(A,C),p1238_2(C,B).
p1238_2(A,B):-p593(A,C),p1260_1(C,B).
p1239(A,B):-p197(A,C),p1239_1(C,B).
p1239_1(A,B):-grab_ball(A,C),p1239_2(C,B).
p1239_2(A,B):-p700(A,C),p164(C,B).
p1249(A,B):-p43_1(A,C),p1249_1(C,B).
p1249_1(A,B):-p979_1(A,C),p410(C,B).
p1251(A,B):-p849(A,C),p1251_1(C,B).
p1251_1(A,B):-p43_1(A,C),p1251_2(C,B).
p1251_2(A,B):-p700(A,C),p552(C,B).
p1263(A,B):-p979_1(A,C),p1263_1(C,B).
p1263_1(A,B):-grab_ball(A,C),p1263_2(C,B).
p1263_2(A,B):-move_right(A,C),drop_ball(C,B).
p1264(A,B):-p254(A,C),p1264_1(C,B).
p1264_1(A,B):-grab_ball(A,C),p1264_2(C,B).
p1264_2(A,B):-p1226(A,C),p153(C,B).
p1276(A,B):-p154(A,C),p647_1(C,B).
p1279(A,B):-move_left(A,C),p1279_1(C,B).
p1279_1(A,B):-p979(A,C),p1279_2(C,B).
p1279_2(A,B):-p43(A,C),p503(C,B).
p1281(A,B):-move_forwards(A,C),p1281_1(C,B).
p1281_1(A,B):-grab_ball(A,C),p1281_2(C,B).
p1281_2(A,B):-move_forwards(A,C),p153(C,B).
p1287(A,B):-move_right(A,C),p1287_1(C,B).
p1287_1(A,B):-p43_1(A,C),p1287_2(C,B).
p1287_2(A,B):-p979_1(A,C),p410(C,B).
p1293(A,B):-p552(A,C),p1293_1(C,B).
p1293_1(A,B):-grab_ball(A,C),p1293_2(C,B).
p1293_2(A,B):-p614(A,C),p406(C,B).
p1295(A,B):-p593(A,C),p1295_1(C,B).
p1295_1(A,B):-grab_ball(A,C),p1295_2(C,B).
p1295_2(A,B):-p979_1(A,C),p1260(C,B).
p1306(A,B):-p43_1(A,C),p1306_1(C,B).
p1306_1(A,B):-p1226(A,C),p1306_2(C,B).
p1306_2(A,B):-p710(A,C),p647_1(C,B).
p1309(A,B):-p979_1(A,C),p1309_1(C,B).
p1309_1(A,B):-grab_ball(A,C),p1309_2(C,B).
p1309_2(A,B):-p110(A,C),p1260_1(C,B).
p1310(A,B):-move_left(A,C),p1310_1(C,B).
p1310_1(A,B):-move_backwards(A,C),p1310_2(C,B).
p1310_2(A,B):-grab_ball(A,C),p507(C,B).
p1316(A,B):-p552(A,C),p593(C,B).
p1319(A,B):-p43_1(A,C),p1319_1(C,B).
p1319_1(A,B):-move_left(A,C),p1319_2(C,B).
p1319_2(A,B):-move_backwards(A,C),p406_1(C,B).
p1323(A,B):-p849(A,C),p1323_1(C,B).
p1323_1(A,B):-grab_ball(A,C),p1323_2(C,B).
p1323_2(A,B):-p507_1(A,C),p979_1(C,B).
p1327(A,B):-p197(A,C),p1327_1(C,B).
p1327_1(A,B):-grab_ball(A,C),p1327_2(C,B).
p1327_2(A,B):-p552_1(A,C),p710(C,B).
p1329(A,B):-move_left(A,C),p1329_1(C,B).
p1329_1(A,B):-grab_ball(A,C),p1329_2(C,B).
p1329_2(A,B):-move_right(A,C),drop_ball(C,B).
p1330(A,B):-p979(A,C),p1330_1(C,B).
p1330_1(A,B):-p43_1(A,C),p1330_2(C,B).
p1330_2(A,B):-move_backwards(A,C),p710(C,B).
p1341(A,B):-p849(A,C),p1341_1(C,B).
p1341_1(A,B):-p43_1(A,C),p1341_2(C,B).
p1341_2(A,B):-p507_1(A,C),p1226(C,B).
p1343(A,B):-move_backwards(A,C),p1343_1(C,B).
p1343_1(A,B):-grab_ball(A,C),p1343_2(C,B).
p1343_2(A,B):-p647_1(A,C),p1260(C,B).
p1344(A,B):-p552_1(A,C),p1344_1(C,B).
p1344_1(A,B):-drop_ball(A,C),p1272(C,B).
p1347(A,B):-p43(A,C),p1347_1(C,B).
p1347_1(A,B):-p552_1(A,C),p1347_2(C,B).
p1347_2(A,B):-p1260(A,C),p193(C,B).
p1352(A,B):-p1226(A,C),p1352_1(C,B).
p1352_1(A,B):-grab_ball(A,C),p1352_2(C,B).
p1352_2(A,B):-p110_1(A,C),p1260_1(C,B).
p1356(A,B):-p979(A,C),p1356_1(C,B).
p1356_1(A,B):-p43(A,C),p1356_2(C,B).
p1356_2(A,B):-p1177_1(A,C),p1260(C,B).
p1368(A,B):-grab_ball(A,C),p1368_1(C,B).
p1368_1(A,B):-p110_1(A,C),p1368_2(C,B).
p1368_2(A,B):-drop_ball(A,C),p979_1(C,B).
p1372(A,B):-p43(A,C),p1372_1(C,B).
p1372_1(A,B):-p1216(A,C),drop_ball(C,B).
p1374(A,B):-p593(A,C),p1374_1(C,B).
p1374_1(A,B):-grab_ball(A,C),p1374_2(C,B).
p1374_2(A,B):-move_right(A,C),p154(C,B).
p1380(A,B):-p1216(A,C),p1380_1(C,B).
p1380_1(A,B):-grab_ball(A,C),p1380_2(C,B).
p1380_2(A,B):-move_backwards(A,C),p406_1(C,B).
p1384(A,B):-p647(A,C),p1384_1(C,B).
p1384_1(A,B):-grab_ball(A,C),p1384_2(C,B).
p1384_2(A,B):-p552_1(A,C),p1260_1(C,B).
p1395(A,B):-p593(A,C),p1395_1(C,B).
p1395_1(A,B):-grab_ball(A,C),p1395_2(C,B).
p1395_2(A,B):-p1226(A,C),p153(C,B).