p87(A,B):-move_right(A,C),move_forwards(C,B).
p116(A,B):-move_right(A,C),move_right(C,B).
p161(A,B):-move_left(A,C),move_right(C,B).
p190(A,B):-move_forwards(A,B).
p273(A,B):-move_backwards(A,C),move_backwards(C,B).
p670(A,B):-move_left(A,C),move_right(C,B).
p793(A,B):-move_forwards(A,C),move_forwards(C,B).
p896(A,B):-move_right(A,C),move_right(C,B).
p1094(A,B):-move_right(A,C),move_right(C,B).
p1242(A,B):-move_right(A,C),move_forwards(C,B).
p1388(A,B):-move_forwards(A,C),move_forwards(C,B).
p1484(A,B):-move_left(A,B).
p76(A,B):-move_backwards(A,C),p273(C,B).
p159(A,B):-move_right(A,C),p159_1(C,B).
p159_1(A,B):-p116(A,C),p273(C,B).
p187(A,B):-p87(A,C),p87(C,B).
p189(A,B):-move_right(A,C),p189_1(C,B).
p189_1(A,B):-p87(A,C),p793(C,B).
p366(A,B):-move_left(A,C),p366_1(C,B).
p366_1(A,B):-move_left(A,C),p793(C,B).
p593(A,B):-grab_ball(A,C),p593_1(C,B).
p593_1(A,B):-move_right(A,C),move_backwards(C,B).
p595(A,B):-p793(A,C),p793(C,B).
p702(A,B):-move_left(A,C),p793(C,B).
p735(A,B):-move_right(A,C),p735_1(C,B).
p735_1(A,B):-p273(A,C),drop_ball(C,B).
p741(A,B):-move_backwards(A,C),p741_1(C,B).
p741_1(A,B):-p116(A,C),p273(C,B).
p742(A,B):-move_backwards(A,C),p742_1(C,B).
p742_1(A,B):-p116(A,C),p273(C,B).
p751(A,B):-move_left(A,C),p273(C,B).
p821(A,B):-move_right(A,C),p821_1(C,B).
p821_1(A,B):-p116(A,C),p273(C,B).
p883(A,B):-p87(A,C),p883_1(C,B).
p883_1(A,B):-p116(A,C),p116(C,B).
p908(A,B):-p87(A,C),p116(C,B).
p912(A,B):-move_left(A,C),p793(C,B).
p914(A,B):-move_forwards(A,C),p793(C,B).
p996(A,B):-move_backwards(A,C),p996_1(C,B).
p996_1(A,B):-p116(A,C),p273(C,B).
p999(A,B):-move_left(A,C),p999_1(C,B).
p999_1(A,B):-move_left(A,C),p273(C,B).
p1067(A,B):-grab_ball(A,C),p1067_1(C,B).
p1067_1(A,B):-move_right(A,C),move_backwards(C,B).
p1100(A,B):-move_forwards(A,C),p793(C,B).
p1116(A,B):-p273(A,C),p273(C,B).
p1166(A,B):-move_left(A,C),p1166_1(C,B).
p1166_1(A,B):-move_left(A,C),move_left(C,B).
p1219(A,B):-move_left(A,C),p1219_1(C,B).
p1219_1(A,B):-move_left(A,C),move_left(C,B).
p1270(A,B):-move_backwards(A,C),p1270_1(C,B).
p1270_1(A,B):-p273(A,C),p273(C,B).
p1283(A,B):-move_right(A,C),p1283_1(C,B).
p1283_1(A,B):-move_backwards(A,C),p116(C,B).
p1286(A,B):-move_left(A,C),p1286_1(C,B).
p1286_1(A,B):-move_left(A,C),move_forwards(C,B).
p1292(A,B):-move_right(A,C),p273(C,B).
p1405(A,B):-move_left(A,C),p1405_1(C,B).
p1405_1(A,B):-move_left(A,C),p793(C,B).
p1536(A,B):-p87(A,C),p1536_1(C,B).
p1536_1(A,B):-p87(A,C),p87(C,B).
p1538(A,B):-move_left(A,C),p273(C,B).
p12(A,B):-p595(A,C),p12_1(C,B).
p12_1(A,B):-p593(A,C),p12_2(C,B).
p12_2(A,B):-drop_ball(A,C),p159_1(C,B).
p42(A,B):-move_left(A,C),p42_1(C,B).
p42_1(A,B):-drop_ball(A,C),p1116(C,B).
p44(A,B):-move_forwards(A,C),p44_1(C,B).
p44_1(A,B):-grab_ball(A,C),p44_2(C,B).
p44_2(A,B):-move_forwards(A,C),drop_ball(C,B).
p89(A,B):-p366_1(A,C),p89_1(C,B).
p89_1(A,B):-grab_ball(A,C),p89_2(C,B).
p89_2(A,B):-p735_1(A,C),p159(C,B).
p109(A,B):-p76(A,C),p109_1(C,B).
p109_1(A,B):-grab_ball(A,C),p109_2(C,B).
p109_2(A,B):-p999(A,C),p1286(C,B).
p111(A,B):-p735_1(A,C),p111_1(C,B).
p111_1(A,B):-p366(A,C),p1166(C,B).
p120(A,B):-p793(A,C),p120_1(C,B).
p120_1(A,B):-drop_ball(A,C),p120_2(C,B).
p120_2(A,B):-move_backwards(A,C),p1166(C,B).
p121(A,B):-p273(A,C),p121_1(C,B).
p121_1(A,B):-grab_ball(A,C),p121_2(C,B).
p121_2(A,B):-move_left(A,C),p1166(C,B).
p132(A,B):-p189_1(A,C),p132_1(C,B).
p132_1(A,B):-p593(A,C),p132_2(C,B).
p132_2(A,B):-p735(A,C),p914(C,B).
p173(A,B):-p116(A,C),p1283(C,B).
p176(A,B):-p1166_1(A,C),p176_1(C,B).
p176_1(A,B):-grab_ball(A,C),p176_2(C,B).
p176_2(A,B):-p735(A,C),move_left(C,B).
p180(A,B):-grab_ball(A,C),p180_1(C,B).
p180_1(A,B):-move_left(A,C),p180_2(C,B).
p180_2(A,B):-drop_ball(A,C),p1166(C,B).
p211(A,B):-p999(A,C),p1286(C,B).
p237(A,B):-grab_ball(A,C),p237_1(C,B).
p237_1(A,B):-p87(A,C),p237_2(C,B).
p237_2(A,B):-drop_ball(A,C),p593_1(C,B).
p246(A,B):-p1166_1(A,C),p246_1(C,B).
p246_1(A,B):-p593(A,C),p246_2(C,B).
p246_2(A,B):-drop_ball(A,C),p87(C,B).
p275(A,B):-p793(A,C),p275_1(C,B).
p275_1(A,B):-grab_ball(A,C),p275_2(C,B).
p275_2(A,B):-p735(A,C),p914(C,B).
p298(A,B):-p273(A,C),p298_1(C,B).
p298_1(A,B):-grab_ball(A,C),p298_2(C,B).
p298_2(A,B):-p735(A,C),p883_1(C,B).
p307(A,B):-p1286(A,C),p307_1(C,B).
p307_1(A,B):-p735_1(A,C),p1536(C,B).
p334(A,B):-p1166(A,C),p334_1(C,B).
p334_1(A,B):-p1286(A,C),p334_2(C,B).
p334_2(A,B):-p593(A,C),drop_ball(C,B).
p335(A,B):-move_left(A,C),p335_1(C,B).
p335_1(A,B):-move_backwards(A,C),p335_2(C,B).
p335_2(A,B):-grab_ball(A,C),p908(C,B).
p338(A,B):-p87(A,C),p338_1(C,B).
p338_1(A,B):-grab_ball(A,C),p338_2(C,B).
p338_2(A,B):-p735_1(A,C),p1166(C,B).
p342(A,B):-p1166(A,C),p342_1(C,B).
p342_1(A,B):-p735_1(A,C),p342_2(C,B).
p342_2(A,B):-move_left(A,C),p187(C,B).
p344(A,B):-p1166_1(A,C),p344_1(C,B).
p344_1(A,B):-p593(A,C),p344_2(C,B).
p344_2(A,B):-drop_ball(A,C),p908(C,B).
p350(A,B):-p908(A,C),p350_1(C,B).
p350_1(A,B):-p593(A,C),p350_2(C,B).
p350_2(A,B):-drop_ball(A,C),p1286_1(C,B).
p365(A,B):-move_forwards(A,C),p365_1(C,B).
p365_1(A,B):-grab_ball(A,C),p365_2(C,B).
p365_2(A,B):-p735_1(A,C),p187(C,B).
p368(A,B):-p366(A,C),p368_1(C,B).
p368_1(A,B):-drop_ball(A,C),p159_1(C,B).
p392(A,B):-move_left(A,C),p392_1(C,B).
p392_1(A,B):-p999(A,C),p1286(C,B).
p458(A,B):-p914(A,C),p458_1(C,B).
p458_1(A,B):-grab_ball(A,C),p458_2(C,B).
p458_2(A,B):-p735(A,C),p1286_1(C,B).
p460(A,B):-move_left(A,C),p460_1(C,B).
p460_1(A,B):-p593(A,C),p460_2(C,B).
p460_2(A,B):-p735_1(A,C),p187(C,B).
p480(A,B):-p187(A,C),p480_1(C,B).
p480_1(A,B):-grab_ball(A,C),p480_2(C,B).
p480_2(A,B):-p735(A,C),p1166_1(C,B).
p488(A,B):-p366_1(A,C),p488_1(C,B).
p488_1(A,B):-drop_ball(A,C),p159(C,B).
p549(A,B):-grab_ball(A,C),p549_1(C,B).
p549_1(A,B):-move_left(A,C),p549_2(C,B).
p549_2(A,B):-drop_ball(A,C),move_backwards(C,B).
p572(A,B):-p914(A,C),p572_1(C,B).
p572_1(A,B):-p593(A,C),p572_2(C,B).
p572_2(A,B):-p735_1(A,C),p1166_1(C,B).
p630(A,B):-p595(A,C),p630_1(C,B).
p630_1(A,B):-grab_ball(A,C),p630_2(C,B).
p630_2(A,B):-p735_1(A,C),p914(C,B).
p655(A,B):-p189_1(A,C),p655_1(C,B).
p655_1(A,B):-p593(A,C),p655_2(C,B).
p655_2(A,B):-p735_1(A,C),p793(C,B).
p658(A,B):-p595(A,C),p658_1(C,B).
p658_1(A,B):-p735(A,C),p751(C,B).
p661(A,B):-p366_1(A,C),p661_1(C,B).
p661_1(A,B):-grab_ball(A,C),p661_2(C,B).
p661_2(A,B):-p735_1(A,C),p189(C,B).
p687(A,B):-grab_ball(A,C),p687_1(C,B).
p687_1(A,B):-p908(A,C),p687_2(C,B).
p687_2(A,B):-drop_ball(A,C),p1116(C,B).
p689(A,B):-move_right(A,C),p689_1(C,B).
p689_1(A,B):-p593(A,C),p689_2(C,B).
p689_2(A,B):-p735(A,C),p914(C,B).
p692(A,B):-p76(A,C),p692_1(C,B).
p692_1(A,B):-p593(A,C),p692_2(C,B).
p692_2(A,B):-drop_ball(A,C),p366(C,B).
p701(A,B):-p189(A,C),p908(C,B).
p736(A,B):-p366(A,C),p736_1(C,B).
p736_1(A,B):-p593(A,C),p736_2(C,B).
p736_2(A,B):-p735(A,C),p1286(C,B).
p744(A,B):-p883_1(A,C),p744_1(C,B).
p744_1(A,B):-drop_ball(A,C),p744_2(C,B).
p744_2(A,B):-p999(A,C),p1286(C,B).
p749(A,B):-p366_1(A,C),p749_1(C,B).
p749_1(A,B):-grab_ball(A,C),p749_2(C,B).
p749_2(A,B):-p735_1(A,C),p593_1(C,B).
p803(A,B):-grab_ball(A,C),p803_1(C,B).
p803_1(A,B):-p187(A,C),p803_2(C,B).
p803_2(A,B):-p735(A,C),move_backwards(C,B).
p818(A,B):-p883_1(A,C),p818_1(C,B).
p818_1(A,B):-grab_ball(A,C),p273(C,B).
p828(A,B):-move_forwards(A,C),p828_1(C,B).
p828_1(A,B):-grab_ball(A,C),p1286_1(C,B).
p843(A,B):-p187(A,C),p843_1(C,B).
p843_1(A,B):-grab_ball(A,C),p366(C,B).
p918(A,B):-move_backwards(A,C),p918_1(C,B).
p918_1(A,B):-p593(A,C),p918_2(C,B).
p918_2(A,B):-p735_1(A,C),p366_1(C,B).
p934(A,B):-p159_1(A,C),p1283(C,B).
p935(A,B):-p116(A,C),p935_1(C,B).
p935_1(A,B):-p593(A,C),p935_2(C,B).
p935_2(A,B):-p735(A,C),p793(C,B).
p939(A,B):-p1292(A,C),p939_1(C,B).
p939_1(A,B):-grab_ball(A,C),p939_2(C,B).
p939_2(A,B):-p735_1(A,C),p914(C,B).
p958(A,B):-p116(A,C),p958_1(C,B).
p958_1(A,B):-grab_ball(A,C),p958_2(C,B).
p958_2(A,B):-p273(A,C),p735(C,B).
p975(A,B):-move_left(A,C),p975_1(C,B).
p975_1(A,B):-grab_ball(A,C),p975_2(C,B).
p975_2(A,B):-p735(A,C),p87(C,B).
p1017(A,B):-p87(A,C),p1017_1(C,B).
p1017_1(A,B):-p735(A,C),p1292(C,B).
p1023(A,B):-move_forwards(A,C),p1023_1(C,B).
p1023_1(A,B):-p999(A,C),p1023_2(C,B).
p1023_2(A,B):-p593(A,C),p914(C,B).
p1047(A,B):-p159(A,C),p1047_1(C,B).
p1047_1(A,B):-grab_ball(A,C),p1047_2(C,B).
p1047_2(A,B):-p76(A,C),p1166(C,B).
p1117(A,B):-p1292(A,C),p1117_1(C,B).
p1117_1(A,B):-p593(A,C),p1117_2(C,B).
p1117_2(A,B):-p735_1(A,C),p366_1(C,B).
p1129(A,B):-move_backwards(A,C),p1129_1(C,B).
p1129_1(A,B):-grab_ball(A,C),p1129_2(C,B).
p1129_2(A,B):-p735_1(A,C),p1166(C,B).
p1137(A,B):-p1166_1(A,C),p1137_1(C,B).
p1137_1(A,B):-p1286(A,C),p593(C,B).
p1152(A,B):-move_right(A,C),p1152_1(C,B).
p1152_1(A,B):-grab_ball(A,C),p1152_2(C,B).
p1152_2(A,B):-p735_1(A,C),p366_1(C,B).
p1168(A,B):-p914(A,C),p1168_1(C,B).
p1168_1(A,B):-grab_ball(A,C),p751(C,B).
p1206(A,B):-p87(A,C),p1206_1(C,B).
p1206_1(A,B):-grab_ball(A,C),p1206_2(C,B).
p1206_2(A,B):-p366_1(A,C),drop_ball(C,B).
p1211(A,B):-p1166(A,C),p1211_1(C,B).
p1211_1(A,B):-drop_ball(A,C),p593_1(C,B).
p1235(A,B):-p1116(A,C),p1235_1(C,B).
p1235_1(A,B):-p593(A,C),p1235_2(C,B).
p1235_2(A,B):-drop_ball(A,C),p914(C,B).
p1238(A,B):-move_left(A,C),p1238_1(C,B).
p1238_1(A,B):-p883(A,C),p1238_2(C,B).
p1238_2(A,B):-grab_ball(A,C),p76(C,B).
p1239(A,B):-p187(A,C),p1239_1(C,B).
p1239_1(A,B):-grab_ball(A,C),p1239_2(C,B).
p1239_2(A,B):-p735_1(A,C),move_forwards(C,B).
p1241(A,B):-p741(A,C),p1241_1(C,B).
p1241_1(A,B):-p593(A,C),p1241_2(C,B).
p1241_2(A,B):-drop_ball(A,C),p793(C,B).
p1275(A,B):-p366(A,C),p1275_1(C,B).
p1275_1(A,B):-p593(A,C),p1275_2(C,B).
p1275_2(A,B):-p735_1(A,C),p1286(C,B).
p1289(A,B):-p1292(A,C),p1289_1(C,B).
p1289_1(A,B):-p593(A,C),p1166_1(C,B).
p1303(A,B):-p1166(A,C),p1286(C,B).
p1327(A,B):-p751(A,C),p999(C,B).
p1329(A,B):-p189(A,C),p1329_1(C,B).
p1329_1(A,B):-p593(A,C),p1329_2(C,B).
p1329_2(A,B):-p735_1(A,C),p116(C,B).
p1348(A,B):-move_right(A,C),p1348_1(C,B).
p1348_1(A,B):-drop_ball(A,C),p1348_2(C,B).
p1348_2(A,B):-p366_1(A,C),p914(C,B).
p1410(A,B):-move_left(A,C),p1410_1(C,B).
p1410_1(A,B):-move_backwards(A,C),p1410_2(C,B).
p1410_2(A,B):-p735_1(A,C),p366(C,B).
p1421(A,B):-p187(A,C),p1421_1(C,B).
p1421_1(A,B):-drop_ball(A,C),p793(C,B).
p1427(A,B):-p793(A,C),p1427_1(C,B).
p1427_1(A,B):-grab_ball(A,C),p1427_2(C,B).
p1427_2(A,B):-p735(A,C),p116(C,B).
p1458(A,B):-p1286(A,C),p1458_1(C,B).
p1458_1(A,B):-p735_1(A,C),p1286_1(C,B).
p1464(A,B):-p189_1(A,C),p1464_1(C,B).
p1464_1(A,B):-grab_ball(A,C),p1464_2(C,B).
p1464_2(A,B):-p735_1(A,C),move_forwards(C,B).
p1466(A,B):-p366_1(A,C),p1466_1(C,B).
p1466_1(A,B):-drop_ball(A,C),p87(C,B).
p1471(A,B):-p751(A,C),p1471_1(C,B).
p1471_1(A,B):-grab_ball(A,C),p1471_2(C,B).
p1471_2(A,B):-p735(A,C),move_forwards(C,B).
p1522(A,B):-p1286_1(A,C),p1522_1(C,B).
p1522_1(A,B):-grab_ball(A,C),p1522_2(C,B).
p1522_2(A,B):-move_right(A,C),p1536(C,B).
p1584(A,B):-p1166(A,C),p1584_1(C,B).
p1584_1(A,B):-grab_ball(A,C),p1584_2(C,B).
p1584_2(A,B):-p116(A,C),drop_ball(C,B).