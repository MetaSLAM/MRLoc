# -*- coding: UTF-8 -*-
from src.util.OrthogonalList import OrthogonalList, ANode, VNode


# def turning_check():
#     if is_turning:
#         newnode = ANode(new_index, None, None)
#         ANodeList.append(newnode)

#         NewVect = VNode(j, current_destination.Data, None, None, 10)
#         if current_destination.Firstout == None:  # 如果是头指针为空,就赋予第一个找到的出度边
#             current_destination.Firstout = NewVect
#             ptail = current_destination.Firstout  # tail指向头
#         else:
#             ptail.Hlink = NewVect
#             ptail = NewVect

# def test_input():
#     item = input_index
#     ol.ShowCrossData(item)

def bioA(ol, local):
    startIs, endIs = ol.CrossMining(local)
    print startIs
    print endIs


