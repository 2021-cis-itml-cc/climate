import os, gzip, time


class file_walker():
    def __init__(self):
        self.INPATH = "data/"
        self.OUTPATH = "data_op/"

        i = 0
        path = []
        name = []
        for sdir, subdir, names in os.walk(self.INPATH):
            i += 1
            if i == 1:
                continue
            else:
                path.append(sdir)
                name.append(names)

        self.dic = {}
        for axes1 in name:
            for axes2 in axes1:
                if axes2 == '.DS_Store':
                    continue
                index = axes2.split('-')
                code1, code2, year = index[0], index[1], index[2]
                # if len(code) == 5 and len(stat) == 6:
                # code, stat = stat, code
                if code1 not in self.dic:
                    self.dic[code1] = [[], [], []]
                self.dic[code1][0].extend([year.split(".")[0]])
                self.dic[code1][1].extend([code2])
                self.dic[code1][2].extend([year.split(".")[0] + "/" + axes2])

        i = 0
        site_gen = self.dic['999999']
        for i in range(len(site_gen[0])):
            code = "b" + site_gen[1][i]
            if code not in self.dic:
                self.dic[code] = [[], [], []]
            self.dic[code][0].extend([site_gen[0][i]])
            self.dic[code][1].extend(["999999"])
            self.dic[code][2].extend([site_gen[2][i]])

        print("Init file_walker Succeed.")
    print("Import file_walker")
    def detect_bsites(self):
        b_site = []
        for i in self.dic.keys():
            if i[0] == 'b':
                b_site.append(i)
        b_site_larger_one = []
        number = []
        for name in b_site:
            if len(self.dic[name][0]) >= 1:
                print(name, len(self.dic[name][0]))
                number.append(len(self.dic[name][0]))
                b_site_larger_one.append(name)
        print(max(number))
        return b_site_larger_one, number

    def max(self):
        keys = self.dic.keys()
        len_list = []
        max = 0
        for key in keys:
            if key == "999999" or key == "949999":
                continue
            if len(self.dic[key][0]) >= max:
                max = len(self.dic[key][0])
                print(max, key)

    def main(self, site_name):
        """

        :param site_name: must in self.dic.keys()
        :return: list, path that is unpacked
        """
        path_op = self.OUTPATH + site_name + "/"
        try:
            os.mkdir(path_op)
        except FileExistsError as e:
            print(e)
        if site_name not in self.dic.keys():
            raise ValueError
        names = self.dic[site_name][2]
        path = []
        for name in names:
            # name = name.replace(".gz", "")
            print(f"[{time.asctime(time.localtime(time.time()))}]:"+self.INPATH + name+" > "+path_op+name.replace(".gz", "").split("/")[1])
            g_file = gzip.GzipFile(self.INPATH + name)
            open(path_op + name.replace(".gz", "").split("/")[1], "wb+").write(g_file.read())
            path.append(path_op + name.replace(".gz", "").split("/")[1])
            g_file.close()
        return path


file_walker()
if __name__ == '__main__':
    file_walker().max()
    file_walker().main("722860")
