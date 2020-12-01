import os
from data import srdata

class Set5(srdata.SRData):
    def __init__(self, args, name='Set5', train=True, benchmark=False):
        if args.data_range==None:
            file_num = len(os.listdir(os.path.join(args.dir_data,'Set5DG/Set5_train_HR'))) * len(args.use_data.split('_'))
            split = str(int(file_num*0.9))
            data_range = [['1', split],[str(int(split)+1), str(file_num)]]
        else:
            data_range = [r.split('-') for r in args.data_range.split('/')]
        
        print("file_num: ",data_range)
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Set5, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(Set5, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(Set5DG, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'Set5_train_HR')
        self.dir_lr = [os.path.join(self.apath, d) for d in self.args.use_data.split("_")]
        if self.input_large: self.dir_lr += 'L'

