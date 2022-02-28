# -*- coding: UTF-8 -*-
class ANode:  # 顶点
    def __init__(self, data, firstin, firstout):
        self.Data = data
        self.Firstin = firstin
        self.Firstout = firstout
        self.index = 0

class VNode:  # 边
    def __init__(self, tailvex, headvex, hlink, tlink, weight):
        self.Tailvex = tailvex
        self.Headvex = headvex
        self.Hlink = hlink
        self.Tlink = tlink
        self.Weight = weight
        self.feature=[]
        self.startI = 2
        self.endI = 3

class OrthogonalList:
    def __init__(self):
        self.W = 4  # 图的边长
        self.INF = float('inf')
        # self.Pic = [[0, 1, 0, 1, 0, 0],
        #        [1, 0, 1, 0, 1, 0],
        #        [0, 1, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 1, 0],
        #        [0, 1, 0, 1, 0, 1],
        #        [0, 0, 1, 0, 1, 0]
        #         ]

        self.Pic = [[0, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 0]]
        self.ANodeList = []

    def InitData(self):
        #还是用index来初始化data吧
        for i in range(self.W):
            newnode = ANode(i,None,None)
            newnode.index = (i+1) * 10
            self.ANodeList.append(newnode)

    def HorizontalLink(self):#横向联系
        for i in range(len(self.ANodeList)):
            ptail = VNode(None,None,None,None,None)
            #搞一个尾指针
            for j in range(self.W):#查找firstout
                if self.Pic[i][j] == 1:
                    print(i, " ", j, " pass")
                    NewVect = VNode(j, self.ANodeList[i].Data, None, None, 10)
                    NewVect.startI = self.ANodeList[i].index
                    NewVect.endI = self.ANodeList[j].index
                    if i > j:
                        NewVect.startI = -NewVect.startI
                        NewVect.endI = -NewVect.endI
                    if self.ANodeList[i].Firstout==None:#如果是头指针为空,就赋予第一个找到的出度边
                        print("build path ", self.ANodeList[i].Data, " to ", j, " from", NewVect.startI, " to ", NewVect.endI)
                        self.ANodeList[i].Firstout = NewVect
                        ptail = self.ANodeList[i].Firstout#tail指向头
                    else:
                        print("build path ", self.ANodeList[i].Data, " to ", j, " from", NewVect.startI, " to ", NewVect.endI)
                        ptail.Hlink = NewVect
                        ptail = NewVect
                # print("一个小循环")
            # print("一个大循环")


    def SearchForVNode(self,head,tail):#这是为了在纵向联系时能找到对应的节点
        for each in self.ANodeList:
            tempVNode = each.Firstout
            while (tempVNode !=None ):
                if tempVNode.Headvex==head and tempVNode.Tailvex==tail:
                    return tempVNode
                else:
                    tempVNode = tempVNode.Hlink

         #正常情况下应该不会返回None
    def VerticalLink(self):#纵向联系
        for i in range(len(self.ANodeList)):
            ptail = VNode(None, None, None, None, None)
            for j in range(self.W):
                if self.Pic[j][i]==1:#这次要竖着来
                    if self.ANodeList[i].Firstin==None:

                        self.ANodeList[i].Firstin=self.SearchForVNode(j,i)
                        ptail = self.ANodeList[i].Firstin
                    else:
                        SearchRes = self.SearchForVNode(j,i)
                        ptail.Tlink = SearchRes
                        ptail = SearchRes

    def ShowCrossData(self, item):#展示十字链表的边，先试试横向的
            ItemFirstOut = item.Firstout
            print('vetex OUT:')
            if ItemFirstOut==None:
                print('None')

            while(ItemFirstOut is not None):
                print(ItemFirstOut.Headvex,'->',ItemFirstOut.Tailvex, ItemFirstOut.startI, ItemFirstOut.endI)
                ItemFirstOut = ItemFirstOut.Hlink
            #下面是纵向输出
            ItemFirstIn  = item.Firstin
            print('vetex IN:')
            if ItemFirstIn==None:
                print('None')

            while(ItemFirstIn is not None):
                print(ItemFirstIn.Headvex,'->',ItemFirstIn.Tailvex, ItemFirstIn.startI, ItemFirstIn.endI)
                ItemFirstIn = ItemFirstIn.Tlink

    def CrossMiningS(self, item):
            startIs, endIs = [], []
            ItemFirstOut = item.Firstout
            print('vetex OUT:')
            if ItemFirstOut==None:
                print('None')

            while(ItemFirstOut is not None):
                print(ItemFirstOut.Headvex,'->',ItemFirstOut.Tailvex)
                endIs.append(ItemFirstOut.endI)
                ItemFirstOut = ItemFirstOut.Hlink


            ItemFirstIn  = item.Firstin
            print('vetex IN:')
            if ItemFirstIn==None:
                print('None')

            while(ItemFirstIn is not None):
                print(ItemFirstIn.Headvex,'->',ItemFirstIn.Tailvex)
                startIs.append(ItemFirstIn.startI)
                ItemFirstIn = ItemFirstIn.Tlink

            return startIs, endIs

    def CrossMining_oldIndex(self, item, item2):
        # if item.Data >= 1:
            startIs, endIs = [], []
            ItemFirstOut = item2.Firstout
            print('vetex OUT:')
            if ItemFirstOut==None:
                print('None')

            while(ItemFirstOut is not None):
                if ItemFirstOut.Tailvex != item.Data:
                    print(ItemFirstOut.Headvex,'->',ItemFirstOut.Tailvex)
                    endIs.append(ItemFirstOut.endI)
                ItemFirstOut = ItemFirstOut.Hlink


            ItemFirstIn  = item.Firstin
            print('vetex IN:')
            if ItemFirstIn==None:
                print('None')

            while(ItemFirstIn is not None):
                if ItemFirstIn.Headvex != item2.Data:
                    print(ItemFirstIn.Headvex,'->',ItemFirstIn.Tailvex)
                    startIs.append(ItemFirstIn.startI)
                ItemFirstIn = ItemFirstIn.Tlink

            return startIs, endIs
        # else:
        #     return self.CrossMiningS(item)

    def CrossMining(self, item, item2):

            startIs, endIs = [], []
            ItemFirstOut = item2.Firstout
            print('vetex OUT:')
            if ItemFirstOut==None:
                print('None')

            while(ItemFirstOut is not None):
                if ItemFirstOut.Tailvex != item.Data:
                    print(ItemFirstOut.Headvex,'->',ItemFirstOut.Tailvex)
                    endIs.append((ItemFirstOut.startI, ItemFirstOut.endI))
                ItemFirstOut = ItemFirstOut.Hlink


            ItemFirstIn  = item.Firstin
            print('vetex IN:')
            if ItemFirstIn==None:
                print('None')

            while(ItemFirstIn is not None):
                if ItemFirstIn.Headvex != item2.Data:
                    print(ItemFirstIn.Headvex,'->',ItemFirstIn.Tailvex)
                    startIs.append((ItemFirstIn.startI, ItemFirstIn.endI))
                ItemFirstIn = ItemFirstIn.Tlink

            core_path = self.SearchForVNode(item.Data, item2.Data)
            core = ((core_path.startI, core_path.endI))

            return startIs, endIs, core

    def map(self):
        # item = self.SearchForVNode(0, 3)
        # item.startI = 1115
        # item.endI = 1205
        # item = self.SearchForVNode(0, 1)
        # item.startI = 6645
        # item.endI = 6815
        # item = self.SearchForVNode(1, 2)
        # item.startI = 2690
        # item.endI = 2965
        # item = self.SearchForVNode(1, 4)
        # item.startI = 0
        # item.endI = 0
        # item = self.SearchForVNode(1, 0)
        # item.startI = 4415
        # item.endI = 4535
        # item = self.SearchForVNode(2, 1)
        # item.startI = 615
        # item.endI = 955
        # item = self.SearchForVNode(2, 5)
        # item.startI = 2965
        # item.endI = 2990
        # item = self.SearchForVNode(3, 0)
        # item.startI = 6560
        # item.endI = 6595
        # item = self.SearchForVNode(3, 4)
        # item.startI = 1205
        # item.endI = 1370
        # item = self.SearchForVNode(4, 1)
        # item.startI = 3390
        # item.endI = 3505
        # item = self.SearchForVNode(4, 3)
        # item.startI = 6929
        # item.endI = 7060
        # item = self.SearchForVNode(4, 5)
        # item.startI = 150
        # item.endI = 485
        # item = self.SearchForVNode(5, 2)
        # item.startI = 485
        # item.endI = 615
        # item = self.SearchForVNode(5, 4)
        # item.startI = 3080
        # item.endI = 3390
        item = self.SearchForVNode(0, 1)
        item.startI = 2351
        item.endI = 2550
        item = self.SearchForVNode(0, 2)
        item.startI = 1468
        item.endI = 1800
        item = self.SearchForVNode(0, 3)
        item.startI = 2246
        item.endI = 2350
        item = self.SearchForVNode(1, 0)
        item.startI = 1251
        item.endI = 1467
        item = self.SearchForVNode(1, 3)
        item.startI = 2551
        item.endI = 2865
        item = self.SearchForVNode(2, 0)
        item.startI = 421
        item.endI = 843
        item = self.SearchForVNode(2, 3)
        item.startI = 1801
        item.endI = 2245
        item = self.SearchForVNode(3, 0)
        item.startI = 844
        item.endI = 982
        item = self.SearchForVNode(3, 1)
        item.startI = 983
        item.endI = 1250
        item = self.SearchForVNode(3, 2)
        item.startI = 1
        item.endI = 420




    def detectPath(self, ref_index):

        for item in self.ANodeList:
            ItemFirstOut = item.Firstout
            while (ItemFirstOut is not None):
                if ItemFirstOut.startI <= ref_index <= ItemFirstOut.endI:
                    index1 = ItemFirstOut.Headvex
                    index2 = ItemFirstOut.Tailvex
                ItemFirstOut = ItemFirstOut.Hlink

        return self.ANodeList[index1], self.ANodeList[index2]


if __name__=='__main__':
    ol = OrthogonalList()
    ol.InitData()
    ol.HorizontalLink()
    ol.VerticalLink()
    ol.map()
    # for item in ol.ANodeList:
    #     print('This is for node%2d:'%(item.Data))
    #     # ol.ShowCrossData(item)
    #     startIs, endIs = ol.CrossMiningS(item)
    #     print startIs
    #     print endIs
    #
    #     print('\n')

    for item in ol.ANodeList:
        print('This is for node%2d-node%1d:'%(item.Data, item.Data+1))

        # ol.ShowCrossData(item)

        # startIs, endIs = ol.CrossMiningS(item)
        # print startIs
        # print endIs

        if item.Data <= ol.W - 2:
            item2 = ol.ANodeList[item.Data+1]
            if item.Data == 1: continue
            startIs, endIs, core = ol.CrossMining(item, item2)
            # print ("Possible start: ", startIs)
            # print ("Possible end: ", endIs)
            for s in startIs:
                for e in endIs:
                    # synax1 = s / abs(s)
                    # synax2 = e / abs(e)
                    # if item < item2:
                    #     synax3 = 1
                    # else:
                    #     synax3 = -1
                    # print("path: ", s, " -> ", synax1*item.index, synax3*item.index, " -> ", synax3*item2.index, synax2*item2.index, " -> ", e)
                    print("path: ", s, " -> ", core, " -> ", e)
        else:
            print("No")
        print('\n')