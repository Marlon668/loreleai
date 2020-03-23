p30(A,B):-move_backwards(A,B).
p40(A,B):-move_right(A,C),move_backwards(C,B).
p232(A,B):-move_left(A,C),move_backwards(C,B).
p567(A,B):-move_left(A,C),move_backwards(C,B).
p630(A,B):-move_left(A,C),move_right(C,B).
p716(A,B):-move_left(A,C),move_right(C,B).
p755(A,B):-move_forwards(A,C),move_forwards(C,B).
p9(A,B):-p9_1(A,C),p9_1(C,B).
p9_1(A,B):-move_right(A,C),move_right(C,B).
p39(A,B):-move_right(A,C),p39_1(C,B).
p39_1(A,B):-p755(A,C),p755(C,B).
p41(A,B):-move_right(A,C),p41_1(C,B).
p41_1(A,B):-move_right(A,C),move_right(C,B).
p109(A,B):-p109_1(A,C),p109_1(C,B).
p109_1(A,B):-move_backwards(A,C),move_backwards(C,B).
p114(A,B):-move_left(A,C),p755(C,B).
p117(A,B):-move_left(A,C),p117_1(C,B).
p117_1(A,B):-p232(A,C),p232(C,B).
p195(A,B):-p232(A,C),p232(C,B).
p278(A,B):-p232(A,C),p232(C,B).
p346(A,B):-move_left(A,C),p346_1(C,B).
p346_1(A,B):-move_forwards(A,C),p755(C,B).
p367(A,B):-p367_1(A,C),p367_1(C,B).
p367_1(A,B):-move_right(A,C),p755(C,B).
p422(A,B):-move_right(A,C),p755(C,B).
p464(A,B):-p464_1(A,C),p464_1(C,B).
p464_1(A,B):-move_left(A,C),p232(C,B).
p601(A,B):-p755(A,C),p601_1(C,B).
p601_1(A,B):-grab_ball(A,C),move_left(C,B).
p42(A,B):-p117(A,C),p42_1(C,B).
p42_1(A,B):-p39_1(A,C),p42_2(C,B).
p42_2(A,B):-drop_ball(A,C),p464_1(C,B).
p60(A,B):-p601(A,C),p60_1(C,B).
p60_1(A,B):-p9_1(A,C),p60_2(C,B).
p60_2(A,B):-drop_ball(A,C),p40(C,B).
p151(A,B):-p232(A,C),p151_1(C,B).
p151_1(A,B):-grab_ball(A,C),p151_2(C,B).
p151_2(A,B):-move_right(A,C),p346_1(C,B).
p154(A,B):-p9(A,C),p154_1(C,B).
p154_1(A,B):-p109(A,C),p154_2(C,B).
p154_2(A,B):-drop_ball(A,C),p346_1(C,B).
p181(A,B):-p117_1(A,C),p181_1(C,B).
p181_1(A,B):-p601_1(A,C),p181_2(C,B).
p181_2(A,B):-drop_ball(A,C),p755(C,B).
p184(A,B):-p232(A,C),p184_1(C,B).
p184_1(A,B):-p601_1(A,C),p184_2(C,B).
p184_2(A,B):-p464_1(A,C),drop_ball(C,B).
p207(A,B):-p39(A,C),p207_1(C,B).
p207_1(A,B):-grab_ball(A,C),p207_2(C,B).
p207_2(A,B):-p117(A,C),p114(C,B).
p251(A,B):-p755(A,C),p251_1(C,B).
p251_1(A,B):-p601(A,C),p251_2(C,B).
p251_2(A,B):-move_backwards(A,C),p9(C,B).
p319(A,B):-p109_1(A,C),p319_1(C,B).
p319_1(A,B):-p601_1(A,C),move_forwards(C,B).
p384(A,B):-p109_1(A,C),p384_1(C,B).
p384_1(A,B):-p117(A,C),p384_2(C,B).
p384_2(A,B):-drop_ball(A,C),p346_1(C,B).
p395(A,B):-p39_1(A,C),p117(C,B).
p424(A,B):-p41(A,C),p424_1(C,B).
p424_1(A,B):-drop_ball(A,C),p424_2(C,B).
p424_2(A,B):-move_left(A,C),p114(C,B).
p427(A,B):-p755(A,C),p427_1(C,B).
p427_1(A,B):-p601(A,C),p427_2(C,B).
p427_2(A,B):-drop_ball(A,C),p109_1(C,B).
p463(A,B):-move_backwards(A,C),p463_1(C,B).
p463_1(A,B):-p601_1(A,C),p463_2(C,B).
p463_2(A,B):-drop_ball(A,C),move_right(C,B).
p487(A,B):-move_right(A,C),p487_1(C,B).
p487_1(A,B):-move_forwards(A,C),p367(C,B).
p525(A,B):-grab_ball(A,C),p525_1(C,B).
p525_1(A,B):-p39_1(A,C),p525_2(C,B).
p525_2(A,B):-drop_ball(A,C),p9(C,B).
p529(A,B):-p39_1(A,C),p529_1(C,B).
p529_1(A,B):-drop_ball(A,C),p529_2(C,B).
p529_2(A,B):-p41(A,C),p109_1(C,B).
p534(A,B):-move_forwards(A,C),p534_1(C,B).
p534_1(A,B):-p9(A,C),p534_2(C,B).
p534_2(A,B):-p601_1(A,C),drop_ball(C,B).
p551(A,B):-move_forwards(A,C),p41(C,B).
p584(A,B):-p464(A,C),p39_1(C,B).
p608(A,B):-move_left(A,C),p346(C,B).
p635(A,B):-move_backwards(A,C),p635_1(C,B).
p635_1(A,B):-drop_ball(A,C),p367_1(C,B).
p691(A,B):-p601(A,C),p691_1(C,B).
p691_1(A,B):-move_backwards(A,C),p691_2(C,B).
p691_2(A,B):-drop_ball(A,C),p109_1(C,B).
p726(A,B):-move_forwards(A,C),p726_1(C,B).
p726_1(A,B):-p109(A,C),p726_2(C,B).
p726_2(A,B):-drop_ball(A,C),p367_1(C,B).
p733(A,B):-move_forwards(A,C),p733_1(C,B).
p733_1(A,B):-p464(A,C),p39_1(C,B).
p737(A,B):-p755(A,C),p737_1(C,B).
p737_1(A,B):-p9(A,C),p737_2(C,B).
p737_2(A,B):-p601(A,C),p117(C,B).
p744(A,B):-move_right(A,C),p744_1(C,B).
p744_1(A,B):-p109_1(A,C),p744_2(C,B).
p744_2(A,B):-grab_ball(A,C),p232(C,B).
p749(A,B):-move_left(A,C),p749_1(C,B).
p749_1(A,B):-p601(A,C),p749_2(C,B).
p749_2(A,B):-p755(A,C),drop_ball(C,B).
p768(A,B):-grab_ball(A,C),p768_1(C,B).
p768_1(A,B):-p755(A,C),p768_2(C,B).
p768_2(A,B):-drop_ball(A,C),p232(C,B).
p779(A,B):-move_right(A,C),p779_1(C,B).
p779_1(A,B):-p601_1(A,C),p779_2(C,B).
p779_2(A,B):-move_forwards(A,C),drop_ball(C,B).
p782(A,B):-p117(A,C),p782_1(C,B).
p782_1(A,B):-p601_1(A,C),p782_2(C,B).
p782_2(A,B):-drop_ball(A,C),p9(C,B).
p791(A,B):-p40(A,C),p791_1(C,B).
p791_1(A,B):-p601(A,C),p791_2(C,B).
p791_2(A,B):-p40(A,C),p40(C,B).