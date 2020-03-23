p174(A,B):-move_right(A,C),move_backwards(C,B).
p293(A,B):-move_right(A,C),move_right(C,B).
p296(A,B):-move_left(A,C),move_left(C,B).
p358(A,B):-move_left(A,C),move_right(C,B).
p421(A,B):-move_right(A,B).
p449(A,B):-move_right(A,C),move_right(C,B).
p518(A,B):-move_right(A,C),move_backwards(C,B).
p11(A,B):-move_left(A,C),p11_1(C,B).
p11_1(A,B):-move_forwards(A,C),move_forwards(C,B).
p36(A,B):-move_left(A,C),p36_1(C,B).
p36_1(A,B):-move_backwards(A,C),move_backwards(C,B).
p39(A,B):-move_right(A,C),p39_1(C,B).
p39_1(A,B):-move_forwards(A,C),move_forwards(C,B).
p99(A,B):-p99_1(A,C),p99_1(C,B).
p99_1(A,B):-move_backwards(A,C),move_backwards(C,B).
p103(A,B):-move_backwards(A,C),p103_1(C,B).
p103_1(A,B):-move_backwards(A,C),p174(C,B).
p375(A,B):-move_right(A,C),p375_1(C,B).
p375_1(A,B):-move_forwards(A,C),move_forwards(C,B).
p428(A,B):-move_right(A,C),p293(C,B).
p435(A,B):-move_left(A,C),p296(C,B).
p508(A,B):-move_left(A,C),p508_1(C,B).
p508_1(A,B):-grab_ball(A,C),p293(C,B).
p509(A,B):-move_backwards(A,C),p509_1(C,B).
p509_1(A,B):-move_backwards(A,C),p174(C,B).
p7(A,B):-p36(A,C),p7_1(C,B).
p7_1(A,B):-p508(A,C),p7_2(C,B).
p7_2(A,B):-drop_ball(A,C),move_forwards(C,B).
p29(A,B):-p296(A,C),p29_1(C,B).
p29_1(A,B):-p11(A,C),p29_2(C,B).
p29_2(A,B):-grab_ball(A,C),p11_1(C,B).
p34(A,B):-p508_1(A,C),p34_1(C,B).
p34_1(A,B):-move_backwards(A,C),p34_2(C,B).
p34_2(A,B):-drop_ball(A,C),p36(C,B).
p40(A,B):-move_left(A,C),p40_1(C,B).
p40_1(A,B):-drop_ball(A,C),p40_2(C,B).
p40_2(A,B):-move_forwards(A,C),p11(C,B).
p43(A,B):-move_left(A,C),p43_1(C,B).
p43_1(A,B):-move_forwards(A,C),p11(C,B).
p62(A,B):-p296(A,C),p62_1(C,B).
p62_1(A,B):-p508(A,C),p62_2(C,B).
p62_2(A,B):-drop_ball(A,C),p11_1(C,B).
p111(A,B):-p508(A,C),p111_1(C,B).
p111_1(A,B):-move_forwards(A,C),p293(C,B).
p130(A,B):-move_right(A,C),p130_1(C,B).
p130_1(A,B):-move_forwards(A,C),p39(C,B).
p153(A,B):-move_right(A,C),p153_1(C,B).
p153_1(A,B):-p508_1(A,C),p153_2(C,B).
p153_2(A,B):-p36_1(A,C),drop_ball(C,B).
p159(A,B):-p36(A,C),p159_1(C,B).
p159_1(A,B):-drop_ball(A,C),p428(C,B).
p188(A,B):-move_forwards(A,C),p188_1(C,B).
p188_1(A,B):-drop_ball(A,C),p188_2(C,B).
p188_2(A,B):-move_forwards(A,C),p435(C,B).
p212(A,B):-p508(A,C),p212_1(C,B).
p212_1(A,B):-p36(A,C),p212_2(C,B).
p212_2(A,B):-drop_ball(A,C),p11(C,B).
p215(A,B):-move_backwards(A,C),p215_1(C,B).
p215_1(A,B):-p508_1(A,C),p215_2(C,B).
p215_2(A,B):-drop_ball(A,C),p39(C,B).
p243(A,B):-move_forwards(A,C),p243_1(C,B).
p243_1(A,B):-p508(A,C),p243_2(C,B).
p243_2(A,B):-drop_ball(A,C),p174(C,B).
p264(A,B):-p508_1(A,C),p264_1(C,B).
p264_1(A,B):-p39(A,C),p264_2(C,B).
p264_2(A,B):-drop_ball(A,C),p103_1(C,B).
p272(A,B):-p11_1(A,C),p11(C,B).
p363(A,B):-p103(A,C),p363_1(C,B).
p363_1(A,B):-p435(A,C),p363_2(C,B).
p363_2(A,B):-drop_ball(A,C),move_right(C,B).
p372(A,B):-p296(A,C),p99(C,B).
p394(A,B):-p293(A,C),p394_1(C,B).
p394_1(A,B):-grab_ball(A,C),p394_2(C,B).
p394_2(A,B):-move_forwards(A,C),drop_ball(C,B).
p415(A,B):-move_forwards(A,C),p415_1(C,B).
p415_1(A,B):-p293(A,C),p39(C,B).
p433(A,B):-move_right(A,C),p433_1(C,B).
p433_1(A,B):-p508_1(A,C),p433_2(C,B).
p433_2(A,B):-p11_1(A,C),drop_ball(C,B).
p438(A,B):-p11_1(A,C),p11(C,B).
p463(A,B):-p428(A,C),p463_1(C,B).
p463_1(A,B):-drop_ball(A,C),p463_2(C,B).
p463_2(A,B):-move_forwards(A,C),p296(C,B).
p481(A,B):-p435(A,C),p481_1(C,B).
p481_1(A,B):-grab_ball(A,C),p481_2(C,B).
p481_2(A,B):-move_backwards(A,C),p36(C,B).
p539(A,B):-move_right(A,C),p539_1(C,B).
p539_1(A,B):-drop_ball(A,C),p36(C,B).
p554(A,B):-p11_1(A,C),p554_1(C,B).
p554_1(A,B):-p11_1(A,C),p554_2(C,B).
p554_2(A,B):-drop_ball(A,C),p428(C,B).
p561(A,B):-grab_ball(A,C),p561_1(C,B).
p561_1(A,B):-p296(A,C),p561_2(C,B).
p561_2(A,B):-drop_ball(A,C),p36(C,B).
p593(A,B):-p428(A,C),p593_1(C,B).
p593_1(A,B):-grab_ball(A,C),p593_2(C,B).
p593_2(A,B):-p36(A,C),drop_ball(C,B).
p596(A,B):-move_forwards(A,C),p596_1(C,B).
p596_1(A,B):-p508(A,C),p596_2(C,B).
p596_2(A,B):-drop_ball(A,C),p174(C,B).