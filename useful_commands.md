# Pymesh

docker run -it -v ~/MVA/GDA/project/Masif/masif:/data pymesh/pymesh

import pymesh

np.save('/data/data/masif_site/output/ground_truth_1A0H_D.npy', pymesh.load_mesh('/data/data/masif_site/data_preparation/01-benchmark_surfaces/1A0H_D.ply').get_attribute('vertex_iface'))


# Masif launch
docker run -it   -v ~/MVA/GDA/project/Masif/masif:/masif/   pablogainza/masif
./data_prepare_one.sh 1A0H_D
./extract_site.sh 1A0H_D