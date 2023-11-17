#!/bin/bash -u

pandir=/media/wentao/Data/Gradient_blending_PAN18/pan18 #./pan18		# Please download PAN18 dataset and save in ./pan18 folder
datadir=./Dataset
Imagedir=./Image
results=./trained 
imagetype=single 	#train Vit by montage multi images as one single or use all singles 	
cachedir=./CACHE
iftrainunimodels=0	# if you want train each uni model, please set it as 1. Otherwise it will only train multimodal models.
LN=en

	
if [ $iftrainunimodels -eq 0 ]; then
    stage=2
    stop_stage=3
    mkdir -p $datadir/$LN
    cp -R ./pretrained/data_pretrained $datadir/$LN/data
    cp -R ./pretrained/cvsplit $datadir/$LN
    GLOWsorucedir=./pretrained/pretrained_results
else
    stage=0
    stop_stage=3
    GLOWsorucedir=$results
fi



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # Read all data and save them in Json files
    if [ -d ${datadir}/$LN ]; then
    	echo "Data alreay processed"
    else
    	echo "Data processing"
    	python3 local/Datapreprocessing.py $pandir $LN $datadir || exit 1
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Train text and image models
    if [ "$(ls -A ${results}/text/model)" ]; then
    	echo "Text model already trained"
    else
    	echo "Training text model"
    	modal=text
    	python3 local/unimodal/train_text.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results 	  		\
    					      --cachedir $cachedir || exit 1
    fi
    if [ -e ${results}/text/results/testsetresults* ]; then
    	echo "Text model alread evaluated"
    else
    	echo "Evaluating text model"
    	modal=text
    	python3 local/unimodal/eval_text.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results 	  		\
    					      --cachedir $cachedir || exit 1
    fi
    if [ -e "$datadir/$LN/errorimagelist.txt" ]; then
    	echo "All images checked"
    else
    	echo "Checking images"
    	python3 local/unimodal/checkimage.py --datasdir $datadir/$LN || exit 1
    fi
    if [ "$(ls -A ${results}/image_single/model)" ]; then
    	echo "Image model already trained"
    else
    	echo "Training Image model"
    	modal=image
    	python3 local/unimodal/train_image.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results 	  		\
    					      --imagetype $imagetype			\
    					      --cachedir $cachedir || exit 1
    fi
    if [ -e ${results}/image_single/results/testsetresults* ]; then
    	echo "Image model alread evaluated"
    else
    	echo "Evaluating image model"
    	modal=image
    	python3 local/unimodal/eval_image.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results 	  		\
    					      --cachedir $cachedir || exit 1
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Train multi-modal models
    
    if [ "$(ls -A ${datadir}/${LN}/data)" ]; then
    	echo "Features already evaluated"
    else
    	echo "extracting features"
    	python3 local/multimodal/feature_extraction.py --datasdir $datadir/$LN 		\
    					      --savedir $results 	  		\
    					      --ifpretrain true			\
    					      --cachedir $cachedir || exit 1
    fi

    if [ "$(ls -A ${results}/dynamic_stream_weight/model)" ]; then
    	echo "DSW model already trained"
    else
    	echo "Training DSW model"
    	modal=dynamic_stream_weight
    	python3 local/multimodal/Dynamic_stream_weighting.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ "$(ls -A ${results}/representation_fusion/model)" ]; then
    	echo "RF model already trained"
    else
    	echo "Training RF model"
    	modal=representation_fusion
    	python3 local/multimodal/train_multi_concate_RF.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ "$(ls -A ${results}/pooling/model)" ]; then
    	echo "pooling model already trained"
    else
    	echo "Training pooling model"
    	modal=pooling
    	python3 local/multimodal/train_pooling.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ "$(ls -A ${Imagedir}/GLOW)" ]; then
    	echo "GLOW model already trained"
    else
    	echo "Training GLOW model"
    	python3 local/multimodal/GLOW.py --sourcedir $GLOWsorucedir 		\
    					      --textmodal text				\
    					      --imagemodal image_single				\
    					      --savedir $Imagedir || exit 1
    fi
    if [ "$(ls -A ${Imagedir}/NLSO)" ]; then
    	echo "NLSO model already trained"
    else
    	echo "Training NLSO model"
    	python3 local/multimodal/NLSO.py --sourcedir $GLOWsorucedir 		\
    					      --textmodal text				\
    					      --imagemodal image_single				\
    					      --savedir $Imagedir || exit 1
    fi
    if [ "$(ls -A ${results}/multi_global_stream_weight/results)" ]; then
    	echo "global stream weighting model already trained"
    else
    	echo "Training global stream weighting model"
    	modal=multi_global_stream_weight
    	python3 local/multimodal/global_stream_weighting.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Train multi-modal models in  sub-files
    # Read all data and save them in Json files
    if [ -e "${datadir}/${LN}/pan18-author-profiling-test-dataset-2018-03-20.json" ]; then
    	echo "Data alreay processed"
    else
    	echo "Data processing"
    	python3 local/Datapreprocessing.py $pandir $LN $datadir || exit 1
    fi
    
    if [ "$(ls -A ${results}/multi_global_stream_weight_subset)" ]; then
    	echo "global stream weighting model in subfiles already trained"
    else
    	echo "Training global stream weighting model in subfiles"
    	modal=multi_global_stream_weight_subset
    	python3 local/Subsetexp/global_stream_weighting.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ "$(ls -A ${results}/GLOW_subset)" ]; then
    	echo "GLOW model in subfiles already trained"
    else
    	echo "Training GLOW model in subfiles"
    	modal=GLOW_subset
    	python3 local/Subsetexp/GLOW.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ "$(ls -A ${results}/NLSO_subset)" ]; then
    	echo "NLSO model in subfiles already trained"
    else
    	echo "Training NLSO model in subfiles"
    	modal=NLSO_subset
    	python3 local/Subsetexp/NLSO.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ -e "${Imagedir}/GLOW-NLSO.pdf" ]; then
    	echo "ploted"
    else
    	echo "Ploting"
    	modal1=GLOW_subset
    	modal2=NLSO_subset
    	python3 local/plot1.py --traindir $results 		\
    			       --savedir $Imagedir				\
    			       --modal1 $modal1 --modal2 $modal2 || exit 1
    fi
    if [ -e "${Imagedir}/GLOW-GSW.pdf" ]; then
    	echo "ploted"
    else
    	echo "Ploting"
    	modal1=GLOW_subset
    	modal2=multi_global_stream_weight_subset
    	python3 local/plot.py --traindir $results 		\
    			       --savedir $Imagedir				\
    			       --modal1 $modal1 --modal2 $modal2 || exit 1
    fi
    if [ "$(ls -A ${results}/dynamic_stream_weight_subset)" ]; then
    	echo "dynamic stream weighting model in subfiles already trained"
    else
    	echo "Training dynamic stream weighting model in subfiles"
    	modal=dynamic_stream_weight_subset
    	python3 local/Subsetexp/Dynamic_stream_weighting.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ -e "${Imagedir}/GLOW-DSW.pdf" ]; then
    	echo "ploted"
    else
    	echo "Ploting"
    	modal1=GLOW_subset
    	modal2=dynamic_stream_weight_subset
    	python3 local/plot.py --traindir $results 		\
    			       --savedir $Imagedir				\
    			       --modal1 $modal1 --modal2 $modal2 || exit 1
    fi
    if [ "$(ls -A ${results}/representation_fusion_subset)" ]; then
    	echo "representation fusion in subfiles already trained"
    else
    	echo "Training representation fusion model in subfiles"
    	modal=representation_fusion_subset
    	python3 local/Subsetexp/train_multi_concate_RF.py --datasdir $datadir/$LN 		\
    					      --modal $modal				\
    					      --savedir $results || exit 1
    fi
    if [ -e "${Imagedir}/GLOW-RF.pdf" ]; then
    	echo "ploted"
    else
    	echo "Ploting"
    	modal1=GLOW_subset
    	modal2=representation_fusion_subset
    	python3 local/plot.py --traindir $results 		\
    			       --savedir $Imagedir				\
    			       --modal1 $modal1 --modal2 $modal2 || exit 1
    fi
fi
