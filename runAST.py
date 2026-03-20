import ast

code = """
def _count_adsorption(self, seg, N, counts, n_vacant_c_sites):

    if n_vacant_c_sites == 0: #no active sites are available on the surface
        return
    if self.n_vacant_h_sites == 0: #no vacant H sites are available on the surface
        return
    if np.any(seg == 1): # no re-adsorption on a fragment that already has adsorbed carbons (competitive Langmuir logic)
        return

    counts['adsorption'][N] += int(np.sum(seg == 0))

"""

tree = ast.parse(code)
print(ast.dump(tree, indent = 2))