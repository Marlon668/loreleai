p17(A,B):-move_right(A,C),move_forwards(C,B).
p32(A,B):-move_left(A,B).
p83(A,B):-move_right(A,B).
p11(A,B):-move_forwards(A,C),p11_1(C,B).
p11_1(A,B):-move_forwards(A,C),move_forwards(C,B).
p87(A,B):-move_right(A,C),p87_1(C,B).
p87_1(A,B):-p17(A,C),p17(C,B).
p92(A,B):-move_forwards(A,C),p92_1(C,B).
p92_1(A,B):-move_forwards(A,C),p17(C,B).
p193(A,B):-move_right(A,C),p17(C,B).
p29(A,B):-move_forwards(A,C),p29_1(C,B).
p29_1(A,B):-grab_ball(A,C),p29_2(C,B).
p29_2(A,B):-move_right(A,C),move_right(C,B).
p55(A,B):-p87_1(A,C),p55_1(C,B).
p55_1(A,B):-grab_ball(A,C),p55_2(C,B).
p55_2(A,B):-p193(A,C),drop_ball(C,B).
p68(A,B):-move_backwards(A,C),p68_1(C,B).
p68_1(A,B):-p68_2(A,C),p68_2(C,B).
p68_2(A,B):-move_right(A,C),move_backwards(C,B).
p110(A,B):-grab_ball(A,C),p110_1(C,B).
p110_1(A,B):-move_backwards(A,C),p110_2(C,B).
p110_2(A,B):-drop_ball(A,C),move_right(C,B).
p122(A,B):-move_left(A,C),p122_1(C,B).
p122_1(A,B):-move_backwards(A,C),p122_2(C,B).
p122_2(A,B):-grab_ball(A,C),move_right(C,B).