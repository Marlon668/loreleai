p22(A,B):-move_left(A,C),move_right(C,B).
p256(A,B):-move_forwards(A,C),move_forwards(C,B).
p272(A,B):-move_right(A,B).
p387(A,B):-move_right(A,B).
p432(A,B):-move_left(A,C),move_forwards(C,B).
p650(A,B):-move_left(A,B).
p653(A,B):-move_right(A,B).
p728(A,B):-move_forwards(A,B).
p833(A,B):-move_left(A,C),move_left(C,B).
p1060(A,B):-move_backwards(A,B).
p1146(A,B):-move_backwards(A,C),move_backwards(C,B).
p1224(A,B):-move_backwards(A,B).
p1373(A,B):-move_left(A,C),move_right(C,B).
p1563(A,B):-move_right(A,C),move_backwards(C,B).
p1928(A,B):-move_backwards(A,C),move_backwards(C,B).
p1941(A,B):-move_right(A,B).
p1952(A,B):-move_forwards(A,B).
p118(A,B):-move_left(A,C),p256(C,B).
p120(A,B):-move_right(A,C),p1563(C,B).
p207(A,B):-p207_1(A,C),p207_1(C,B).
p207_1(A,B):-p833(A,C),p1146(C,B).
p255(A,B):-p255_1(A,C),p255_1(C,B).
p255_1(A,B):-p833(A,C),p1146(C,B).
p265(A,B):-move_right(A,C),p265_1(C,B).
p265_1(A,B):-move_forwards(A,C),p256(C,B).
p403(A,B):-move_left(A,C),p403_1(C,B).
p403_1(A,B):-p256(A,C),p256(C,B).
p414(A,B):-move_left(A,C),p414_1(C,B).
p414_1(A,B):-move_backwards(A,C),p833(C,B).
p586(A,B):-move_forwards(A,C),p586_1(C,B).
p586_1(A,B):-p256(A,C),p256(C,B).
p593(A,B):-move_forwards(A,C),p256(C,B).
p601(A,B):-move_left(A,C),p833(C,B).
p646(A,B):-p256(A,C),p646_1(C,B).
p646_1(A,B):-p833(A,C),p833(C,B).
p673(A,B):-move_left(A,C),p673_1(C,B).
p673_1(A,B):-p833(A,C),p1146(C,B).
p777(A,B):-move_left(A,C),p777_1(C,B).
p777_1(A,B):-move_backwards(A,C),p833(C,B).
p783(A,B):-move_left(A,C),p783_1(C,B).
p783_1(A,B):-move_backwards(A,C),p1146(C,B).
p837(A,B):-move_right(A,C),p1146(C,B).
p914(A,B):-p256(A,C),p256(C,B).
p929(A,B):-move_right(A,C),p1146(C,B).
p944(A,B):-p1146(A,C),p1563(C,B).
p1024(A,B):-move_backwards(A,C),p1146(C,B).
p1028(A,B):-move_right(A,C),p1028_1(C,B).
p1028_1(A,B):-move_right(A,C),move_forwards(C,B).
p1073(A,B):-p256(A,C),p256(C,B).
p1113(A,B):-move_left(A,C),p833(C,B).
p1130(A,B):-move_left(A,C),p1130_1(C,B).
p1130_1(A,B):-p256(A,C),p833(C,B).
p1279(A,B):-move_backwards(A,C),p1279_1(C,B).
p1279_1(A,B):-drop_ball(A,C),p432(C,B).
p1337(A,B):-p833(A,C),p1146(C,B).
p1341(A,B):-move_right(A,C),p1341_1(C,B).
p1341_1(A,B):-move_right(A,C),move_right(C,B).
p1405(A,B):-p256(A,C),p833(C,B).
p1544(A,B):-p256(A,C),p1544_1(C,B).
p1544_1(A,B):-p432(A,C),p833(C,B).
p1575(A,B):-move_backwards(A,C),p1575_1(C,B).
p1575_1(A,B):-p833(A,C),p1146(C,B).
p1639(A,B):-move_left(A,C),p833(C,B).
p1784(A,B):-move_backwards(A,C),p1784_1(C,B).
p1784_1(A,B):-p833(A,C),p833(C,B).
p1828(A,B):-move_left(A,C),p1828_1(C,B).
p1828_1(A,B):-move_backwards(A,C),p1146(C,B).
p1877(A,B):-move_right(A,C),p1563(C,B).
p1878(A,B):-p1146(A,C),p1146(C,B).
p1911(A,B):-p1563(A,C),p1911_1(C,B).
p1911_1(A,B):-p1563(A,C),p1563(C,B).
p1920(A,B):-move_backwards(A,C),p1920_1(C,B).
p1920_1(A,B):-p833(A,C),p833(C,B).
p1987(A,B):-p256(A,C),p833(C,B).
p51(A,B):-p118(A,C),p51_1(C,B).
p51_1(A,B):-p1279_1(A,C),p783_1(C,B).
p64(A,B):-p1784(A,C),p64_1(C,B).
p64_1(A,B):-grab_ball(A,C),p64_2(C,B).
p64_2(A,B):-p1279(A,C),p944(C,B).
p77(A,B):-p833(A,C),p77_1(C,B).
p77_1(A,B):-grab_ball(A,C),p77_2(C,B).
p77_2(A,B):-move_right(A,C),p1279(C,B).
p83(A,B):-move_right(A,C),p83_1(C,B).
p83_1(A,B):-p1028(A,C),p83_2(C,B).
p83_2(A,B):-p1279_1(A,C),p256(C,B).
p122(A,B):-p118(A,C),p122_1(C,B).
p122_1(A,B):-drop_ball(A,C),p122_2(C,B).
p122_2(A,B):-p1341_1(A,C),p1341(C,B).
p147(A,B):-p265(A,C),p1341(C,B).
p152(A,B):-p1341_1(A,C),p152_1(C,B).
p152_1(A,B):-grab_ball(A,C),p152_2(C,B).
p152_2(A,B):-p833(A,C),p1279(C,B).
p208(A,B):-p265_1(A,C),p208_1(C,B).
p208_1(A,B):-p1341(A,C),p208_2(C,B).
p208_2(A,B):-drop_ball(A,C),move_backwards(C,B).
p211(A,B):-p256(A,C),p211_1(C,B).
p211_1(A,B):-grab_ball(A,C),p211_2(C,B).
p211_2(A,B):-p1279(A,C),move_left(C,B).
p218(A,B):-p414(A,C),p218_1(C,B).
p218_1(A,B):-drop_ball(A,C),p218_2(C,B).
p218_2(A,B):-p256(A,C),p1341(C,B).
p222(A,B):-move_right(A,C),p1911(C,B).
p224(A,B):-p414_1(A,C),p224_1(C,B).
p224_1(A,B):-grab_ball(A,C),p224_2(C,B).
p224_2(A,B):-p1279(A,C),p256(C,B).
p243(A,B):-move_left(A,C),p243_1(C,B).
p243_1(A,B):-p1575(A,C),p243_2(C,B).
p243_2(A,B):-drop_ball(A,C),p1341_1(C,B).
p246(A,B):-move_left(A,C),p246_1(C,B).
p246_1(A,B):-p432(A,C),p246_2(C,B).
p246_2(A,B):-grab_ball(A,C),move_backwards(C,B).
p279(A,B):-p432(A,C),p279_1(C,B).
p279_1(A,B):-grab_ball(A,C),p279_2(C,B).
p279_2(A,B):-p1341(A,C),p1279_1(C,B).
p313(A,B):-p207_1(A,C),p313_1(C,B).
p313_1(A,B):-grab_ball(A,C),p313_2(C,B).
p313_2(A,B):-p1028(A,C),p1279_1(C,B).
p367(A,B):-p1130_1(A,C),p367_1(C,B).
p367_1(A,B):-drop_ball(A,C),p1341_1(C,B).
p368(A,B):-p120(A,C),p1911(C,B).
p377(A,B):-p783(A,C),p377_1(C,B).
p377_1(A,B):-grab_ball(A,C),p377_2(C,B).
p377_2(A,B):-p1028(A,C),p1279(C,B).
p380(A,B):-move_left(A,C),p380_1(C,B).
p380_1(A,B):-grab_ball(A,C),p1130_1(C,B).
p404(A,B):-p1028(A,C),p404_1(C,B).
p404_1(A,B):-p1341(A,C),p404_2(C,B).
p404_2(A,B):-p1279(A,C),p837(C,B).
p415(A,B):-p646(A,C),p415_1(C,B).
p415_1(A,B):-grab_ball(A,C),p415_2(C,B).
p415_2(A,B):-p1146(A,C),drop_ball(C,B).
p440(A,B):-move_left(A,C),p440_1(C,B).
p440_1(A,B):-p1878(A,C),p440_2(C,B).
p440_2(A,B):-drop_ball(A,C),p1341_1(C,B).
p441(A,B):-move_forwards(A,C),p441_1(C,B).
p441_1(A,B):-grab_ball(A,C),p441_2(C,B).
p441_2(A,B):-p1146(A,C),p944(C,B).
p454(A,B):-p1130_1(A,C),p454_1(C,B).
p454_1(A,B):-grab_ball(A,C),p454_2(C,B).
p454_2(A,B):-move_right(A,C),p1279_1(C,B).
p459(A,B):-p1028_1(A,C),p459_1(C,B).
p459_1(A,B):-drop_ball(A,C),p1028(C,B).
p505(A,B):-grab_ball(A,C),p505_1(C,B).
p505_1(A,B):-move_right(A,C),p505_2(C,B).
p505_2(A,B):-p1279_1(A,C),p1146(C,B).
p565(A,B):-p837(A,C),p565_1(C,B).
p565_1(A,B):-grab_ball(A,C),p565_2(C,B).
p565_2(A,B):-p1279(A,C),p783_1(C,B).
p579(A,B):-p833(A,C),p579_1(C,B).
p579_1(A,B):-grab_ball(A,C),p579_2(C,B).
p579_2(A,B):-p1279(A,C),p1341(C,B).
p620(A,B):-p586(A,C),p620_1(C,B).
p620_1(A,B):-grab_ball(A,C),p620_2(C,B).
p620_2(A,B):-p1279(A,C),p120(C,B).
p647(A,B):-move_left(A,C),p647_1(C,B).
p647_1(A,B):-p1911(A,C),p647_2(C,B).
p647_2(A,B):-p1279(A,C),p1341(C,B).
p648(A,B):-p1341_1(A,C),p648_1(C,B).
p648_1(A,B):-drop_ball(A,C),p944(C,B).
p666(A,B):-p1146(A,C),p666_1(C,B).
p666_1(A,B):-grab_ball(A,C),p666_2(C,B).
p666_2(A,B):-p1279(A,C),p1028_1(C,B).
p683(A,B):-grab_ball(A,C),p683_1(C,B).
p683_1(A,B):-p120(A,C),p683_2(C,B).
p683_2(A,B):-drop_ball(A,C),p256(C,B).
p689(A,B):-p414(A,C),p689_1(C,B).
p689_1(A,B):-grab_ball(A,C),p689_2(C,B).
p689_2(A,B):-p1341(A,C),drop_ball(C,B).
p708(A,B):-p783(A,C),p708_1(C,B).
p708_1(A,B):-grab_ball(A,C),p708_2(C,B).
p708_2(A,B):-p1544_1(A,C),drop_ball(C,B).
p738(A,B):-p414_1(A,C),p738_1(C,B).
p738_1(A,B):-drop_ball(A,C),p833(C,B).
p741(A,B):-p265(A,C),p741_1(C,B).
p741_1(A,B):-p1341(A,C),p741_2(C,B).
p741_2(A,B):-grab_ball(A,C),p646_1(C,B).
p745(A,B):-p833(A,C),p745_1(C,B).
p745_1(A,B):-grab_ball(A,C),p745_2(C,B).
p745_2(A,B):-move_right(A,C),p403_1(C,B).
p751(A,B):-p1028_1(A,C),p751_1(C,B).
p751_1(A,B):-drop_ball(A,C),p751_2(C,B).
p751_2(A,B):-move_left(A,C),p1784(C,B).
p753(A,B):-p1911(A,C),p753_1(C,B).
p753_1(A,B):-p1279(A,C),p753_2(C,B).
p753_2(A,B):-move_left(A,C),p403(C,B).
p759(A,B):-p1563(A,C),p759_1(C,B).
p759_1(A,B):-grab_ball(A,C),p759_2(C,B).
p759_2(A,B):-p673(A,C),drop_ball(C,B).
p857(A,B):-p1130(A,C),p857_1(C,B).
p857_1(A,B):-grab_ball(A,C),p857_2(C,B).
p857_2(A,B):-p1341(A,C),p1279(C,B).
p864(A,B):-p833(A,C),p864_1(C,B).
p864_1(A,B):-grab_ball(A,C),p864_2(C,B).
p864_2(A,B):-p837(A,C),p1279(C,B).
p878(A,B):-p601(A,C),p878_1(C,B).
p878_1(A,B):-grab_ball(A,C),p878_2(C,B).
p878_2(A,B):-move_right(A,C),p1544(C,B).
p879(A,B):-grab_ball(A,C),p879_1(C,B).
p879_1(A,B):-p1279(A,C),move_backwards(C,B).
p888(A,B):-grab_ball(A,C),p888_1(C,B).
p888_1(A,B):-p1279(A,C),p1575(C,B).
p905(A,B):-p1911_1(A,C),p905_1(C,B).
p905_1(A,B):-grab_ball(A,C),p905_2(C,B).
p905_2(A,B):-move_right(A,C),p256(C,B).
p932(A,B):-p265(A,C),p932_1(C,B).
p932_1(A,B):-grab_ball(A,C),p1544_1(C,B).
p937(A,B):-p1544_1(A,C),p937_1(C,B).
p937_1(A,B):-grab_ball(A,C),p937_2(C,B).
p937_2(A,B):-p783(A,C),p1279_1(C,B).
p957(A,B):-move_right(A,C),p957_1(C,B).
p957_1(A,B):-p1544(A,C),p957_2(C,B).
p957_2(A,B):-grab_ball(A,C),p1279(C,B).
p977(A,B):-move_right(A,C),p977_1(C,B).
p977_1(A,B):-grab_ball(A,C),p977_2(C,B).
p977_2(A,B):-move_right(A,C),p1279(C,B).
p980(A,B):-move_backwards(A,C),p980_1(C,B).
p980_1(A,B):-grab_ball(A,C),p980_2(C,B).
p980_2(A,B):-move_backwards(A,C),drop_ball(C,B).
p1007(A,B):-p120(A,C),p1007_1(C,B).
p1007_1(A,B):-p1279_1(A,C),p1130(C,B).
p1009(A,B):-p1911(A,C),p1009_1(C,B).
p1009_1(A,B):-grab_ball(A,C),p1009_2(C,B).
p1009_2(A,B):-p1279(A,C),p1341_1(C,B).
p1040(A,B):-p1028(A,C),p1040_1(C,B).
p1040_1(A,B):-p1028(A,C),p1040_2(C,B).
p1040_2(A,B):-grab_ball(A,C),p1784(C,B).
p1069(A,B):-p118(A,C),p1069_1(C,B).
p1069_1(A,B):-grab_ball(A,C),p1069_2(C,B).
p1069_2(A,B):-p944(A,C),p1279(C,B).
p1077(A,B):-p1028(A,C),p1077_1(C,B).
p1077_1(A,B):-grab_ball(A,C),p1077_2(C,B).
p1077_2(A,B):-p1279(A,C),p207_1(C,B).
p1119(A,B):-move_backwards(A,C),p1119_1(C,B).
p1119_1(A,B):-grab_ball(A,C),p1119_2(C,B).
p1119_2(A,B):-p783(A,C),drop_ball(C,B).
p1126(A,B):-p118(A,C),p646(C,B).
p1127(A,B):-move_right(A,C),p265(C,B).
p1139(A,B):-grab_ball(A,C),p1139_1(C,B).
p1139_1(A,B):-move_backwards(A,C),p1139_2(C,B).
p1139_2(A,B):-drop_ball(A,C),p833(C,B).
p1142(A,B):-grab_ball(A,C),p1142_1(C,B).
p1142_1(A,B):-p403_1(A,C),p1142_2(C,B).
p1142_2(A,B):-drop_ball(A,C),p1911_1(C,B).
p1147(A,B):-move_backwards(A,C),p1147_1(C,B).
p1147_1(A,B):-grab_ball(A,C),p207_1(C,B).
p1159(A,B):-move_left(A,C),p1159_1(C,B).
p1159_1(A,B):-grab_ball(A,C),p1130(C,B).
p1167(A,B):-p414(A,C),p1167_1(C,B).
p1167_1(A,B):-p1279_1(A,C),p1146(C,B).
p1181(A,B):-move_left(A,C),p1575(C,B).
p1197(A,B):-p833(A,C),p1197_1(C,B).
p1197_1(A,B):-grab_ball(A,C),p1197_2(C,B).
p1197_2(A,B):-p1130_1(A,C),p1279_1(C,B).
p1203(A,B):-p1130_1(A,C),p1203_1(C,B).
p1203_1(A,B):-grab_ball(A,C),p1203_2(C,B).
p1203_2(A,B):-p1341(A,C),p1279_1(C,B).
p1230(A,B):-p646_1(A,C),p1230_1(C,B).
p1230_1(A,B):-grab_ball(A,C),p120(C,B).
p1273(A,B):-p120(A,C),p1878(C,B).
p1275(A,B):-move_right(A,C),p1275_1(C,B).
p1275_1(A,B):-drop_ball(A,C),p120(C,B).
p1303(A,B):-move_left(A,C),p1303_1(C,B).
p1303_1(A,B):-move_backwards(A,C),p1303_2(C,B).
p1303_2(A,B):-grab_ball(A,C),p837(C,B).
p1307(A,B):-p1544_1(A,C),p1307_1(C,B).
p1307_1(A,B):-grab_ball(A,C),p1575(C,B).
p1310(A,B):-p403(A,C),p1310_1(C,B).
p1310_1(A,B):-grab_ball(A,C),p1310_2(C,B).
p1310_2(A,B):-p1279(A,C),p1575(C,B).
p1316(A,B):-p207_1(A,C),p1316_1(C,B).
p1316_1(A,B):-grab_ball(A,C),p1316_2(C,B).
p1316_2(A,B):-p1279(A,C),move_left(C,B).
p1321(A,B):-p1563(A,C),p1321_1(C,B).
p1321_1(A,B):-p1279(A,C),p432(C,B).
p1327(A,B):-p1911_1(A,C),p1327_1(C,B).
p1327_1(A,B):-grab_ball(A,C),p1327_2(C,B).
p1327_2(A,B):-p1130_1(A,C),p1279_1(C,B).
p1330(A,B):-move_right(A,C),p1330_1(C,B).
p1330_1(A,B):-p1544(A,C),p1330_2(C,B).
p1330_2(A,B):-drop_ball(A,C),p120(C,B).
p1361(A,B):-p1341(A,C),p1361_1(C,B).
p1361_1(A,B):-grab_ball(A,C),p1361_2(C,B).
p1361_2(A,B):-p1028_1(A,C),p1279_1(C,B).
p1381(A,B):-grab_ball(A,C),p1381_1(C,B).
p1381_1(A,B):-move_right(A,C),p1381_2(C,B).
p1381_2(A,B):-p265(A,C),p1279(C,B).
p1387(A,B):-p120(A,C),p1387_1(C,B).
p1387_1(A,B):-p1279(A,C),p833(C,B).
p1418(A,B):-p601(A,C),p1418_1(C,B).
p1418_1(A,B):-grab_ball(A,C),p1418_2(C,B).
p1418_2(A,B):-move_backwards(A,C),p1279(C,B).
p1477(A,B):-p403(A,C),p1477_1(C,B).
p1477_1(A,B):-grab_ball(A,C),p1477_2(C,B).
p1477_2(A,B):-p120(A,C),p120(C,B).
p1485(A,B):-p265(A,C),p1485_1(C,B).
p1485_1(A,B):-p1279(A,C),p1485_2(C,B).
p1485_2(A,B):-move_backwards(A,C),p1911(C,B).
p1507(A,B):-p403(A,C),p1507_1(C,B).
p1507_1(A,B):-p1279(A,C),p1341_1(C,B).
p1511(A,B):-p1028_1(A,C),p1511_1(C,B).
p1511_1(A,B):-grab_ball(A,C),p1511_2(C,B).
p1511_2(A,B):-p1279(A,C),p1028(C,B).
p1560(A,B):-move_right(A,C),p207(C,B).
p1562(A,B):-p1028_1(A,C),p1562_1(C,B).
p1562_1(A,B):-grab_ball(A,C),p1562_2(C,B).
p1562_2(A,B):-move_forwards(A,C),drop_ball(C,B).
p1583(A,B):-p256(A,C),p1544(C,B).
p1604(A,B):-p432(A,C),p1604_1(C,B).
p1604_1(A,B):-grab_ball(A,C),p1604_2(C,B).
p1604_2(A,B):-move_left(A,C),p1146(C,B).
p1619(A,B):-p783(A,C),p1619_1(C,B).
p1619_1(A,B):-grab_ball(A,C),p1619_2(C,B).
p1619_2(A,B):-move_right(A,C),p265(C,B).
p1623(A,B):-p120(A,C),p1623_1(C,B).
p1623_1(A,B):-grab_ball(A,C),p1623_2(C,B).
p1623_2(A,B):-p1279(A,C),p207_1(C,B).
p1669(A,B):-p1563(A,C),p1669_1(C,B).
p1669_1(A,B):-grab_ball(A,C),p1669_2(C,B).
p1669_2(A,B):-p1279(A,C),p1028(C,B).
p1695(A,B):-grab_ball(A,C),p1695_1(C,B).
p1695_1(A,B):-p1130_1(A,C),p1695_2(C,B).
p1695_2(A,B):-p1279(A,C),p1911_1(C,B).
p1702(A,B):-p1575(A,C),p1702_1(C,B).
p1702_1(A,B):-p1279(A,C),move_forwards(C,B).
p1722(A,B):-p1146(A,C),p1722_1(C,B).
p1722_1(A,B):-grab_ball(A,C),p1722_2(C,B).
p1722_2(A,B):-p1279(A,C),p601(C,B).
p1824(A,B):-p1028(A,C),p1341(C,B).
p1825(A,B):-p120(A,C),p1341(C,B).
p1884(A,B):-p833(A,C),p1884_1(C,B).
p1884_1(A,B):-grab_ball(A,C),p1884_2(C,B).
p1884_2(A,B):-move_right(A,C),p1028(C,B).
p1889(A,B):-move_forwards(A,C),p1889_1(C,B).
p1889_1(A,B):-p1544(A,C),p1889_2(C,B).
p1889_2(A,B):-grab_ball(A,C),p1563(C,B).
p1940(A,B):-p783(A,C),p1940_1(C,B).
p1940_1(A,B):-grab_ball(A,C),move_backwards(C,B).
p1942(A,B):-move_left(A,C),p1942_1(C,B).
p1942_1(A,B):-grab_ball(A,C),p1942_2(C,B).
p1942_2(A,B):-p1279(A,C),p403_1(C,B).
p1975(A,B):-p1341_1(A,C),p1975_1(C,B).
p1975_1(A,B):-grab_ball(A,C),p1784(C,B).
p1979(A,B):-p1341_1(A,C),p1979_1(C,B).
p1979_1(A,B):-grab_ball(A,C),p1130(C,B).
p1990(A,B):-move_left(A,C),p1990_1(C,B).
p1990_1(A,B):-p646_1(A,C),p1990_2(C,B).
p1990_2(A,B):-grab_ball(A,C),move_backwards(C,B).
p1993(A,B):-grab_ball(A,C),p1993_1(C,B).
p1993_1(A,B):-move_forwards(A,C),p1993_2(C,B).
p1993_2(A,B):-drop_ball(A,C),p1028_1(C,B).
p1994(A,B):-p783(A,C),p1994_1(C,B).
p1994_1(A,B):-grab_ball(A,C),p1994_2(C,B).
p1994_2(A,B):-p1028_1(A,C),p1279_1(C,B).