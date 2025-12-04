import sys
import os
import numpy as np
import tensorflow as tf
from masif_modules.MaSIF_site import MaSIF_site
from default_config.masif_opts import masif_opts
import importlib

# Configuration
def mask_input_feat(input_feat, mask):
    mymask = np.where(np.array(mask) == 0.0)[0]
    return np.delete(input_feat, mymask, axis=2)


params = masif_opts["site"]
custom_params_file = sys.argv[1]
custom_params = importlib.import_module(custom_params_file, package=None)
custom_params = custom_params.custom_params

for key in custom_params:
    print("Setting {} to {} ".format(key, custom_params[key]))
    params[key] = custom_params[key]


# Shape precomputation dir.
parent_in_dir = params["masif_precomputation_dir"]
eval_list = []

if len(sys.argv) == 3:
    ppi_pair_ids = [sys.argv[2]]
# Read a list of pdb_chain entries to evaluate.
elif len(sys.argv) == 4 and sys.argv[2] == "-l":
    listfile = open(sys.argv[3])
    ppi_pair_ids = []
    for line in listfile:
        eval_list.append(line.rstrip())
    for mydir in os.listdir(parent_in_dir):
        ppi_pair_ids.append(mydir)
else:
    sys.exit(1)

# Initialisation du graphe TF
with tf.Graph().as_default():
    # Instanciation du modèle MaSIF-site
    model = MaSIF_site(
        params["max_distance"],
        n_thetas=4,
        n_rhos=3,
        n_rotations=4,
        idx_gpu="/gpu:0",
        feat_mask=params["feat_mask"],
        n_conv_layers=params["n_conv_layers"],
    )

    # Le tenseur contenant les embeddings est généralement 'model.global_desc'
    # ou accessible via la dernière couche de convolution avant le MLP final.
    # Dans MaSIF-site, 'global_desc' est le descripteur du patch.
    embedding_tensor = model.global_desc 
    score_tensor = model.score

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Chargement des poids du modèle
        ckpt = tf.train.get_checkpoint_state(params['model_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Modèle chargé depuis: %s" % ckpt.model_checkpoint_path)
        else:
            print("Erreur: Modèle non trouvé.")
            sys.exit(1)

        print(f"Extraction des embeddings pour {target_pdb_id}...")
        
        # Chargement des données du patch pour la protéine cible
        try:
            # Note: Cette fonction charge les batchs. Assurez-vous que le chemin est correct.
            # masif_site stocke les données pré-calculées dans des structures spécifiques.
            # Nous itérons manuellement si nécessaire, ou utilisons le générateur.
            # Pour simplifier, on suppose que read_data... gère l'itération ou on charge tout.
            
            # Pour l'inférence simple, on charge souvent via une liste
            list_of_pdbs = [target_pdb_id]
            
            # Boucle sur les données (simulé ici comme dans masif_site_predict.py)
            # Vous devrez peut-être adapter 'read_data_from_matfile...' selon votre version exacte du repo
            # L'idée est d'obtenir rho_coords, theta_coords, mask, input_feat pour chaque batch.
            
            # Exemple simplifié inspiré de predict_site.py:
            # On charge tout le fichier precomputé
            mydir = parent_in_dir + target_pdb_id + '/'
            rho_w_name = mydir + target_pdb_id + '_rho_wcoords.npy'
            theta_w_name = mydir + target_pdb_id + '_theta_wcoords.npy'
            input_feat_name = mydir + target_pdb_id + '_input_feat.npy'
            mask_name = mydir + target_pdb_id + '_mask.npy'
            
            if not os.path.exists(rho_w_name):
                print(f"Fichiers manquants pour {target_pdb_id}. Avez-vous lancé data_prepare_one.sh ?")
                sys.exit(1)
                
            rho_w = np.load(rho_w_name)
            theta_w = np.load(theta_w_name)
            input_feat = np.load(input_feat_name)
            mask = np.load(mask_name)
            
            num_patches = len(rho_w)
            all_embeddings = []
            
            # Inférence par batch
            batch_size = params['batch_size']
            for i in range(0, num_patches, batch_size):
                end_i = min(i + batch_size, num_patches)
                feed_dict = {
                    model.rho_coords: rho_w[i:end_i],
                    model.theta_coords: theta_w[i:end_i],
                    model.input_feat: input_feat[i:end_i],
                    model.mask: mask[i:end_i],
                    model.keep_prob: 1.0 # Pas de dropout en inférence
                }
                
                # C'est ici qu'on récupère les embeddings !
                emb_batch = sess.run(embedding_tensor, feed_dict=feed_dict)
                all_embeddings.append(emb_batch)
                
            # Concatenate et sauvegarde
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            
            save_path = os.path.join(out_dir, f"{target_pdb_id}_embeddings.npy")
            np.save(save_path, final_embeddings)
            
            print(f"Succès ! Embeddings sauvegardés dans : {save_path}")
            print(f"Forme du tenseur : {final_embeddings.shape}") # Devrait être (N_patchs, 32) ou (N_patchs, 80) selon config
            
        except Exception as e:
            print(f"Erreur lors de l'extraction: {e}")