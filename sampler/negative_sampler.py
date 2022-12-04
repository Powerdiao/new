import dgl


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes
        }
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            n = len(src)
            if self.neg_share and n % self.k == 0:
                dst = self.weights[etype].multinomial(n, replacement=True)
                dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
            else:
                dst = self.weights[etype].multinomial(n*self.k, replacement=True)
            src = src.repeat_interleave(self.k)
            result_dict[etype] = (src, dst)
            # print(etype, src.shape, dst.shape)

        # print([(k, v[-1].shape) for k, v in result_dict.items()])
        return result_dict