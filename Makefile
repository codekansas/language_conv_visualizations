# Helps batch some common operations.

LANG_ENV_NAME=lang.env
PROD_ENV_NAME=prod.env

_lang_env: ${LANG_ENV_NAME}/bin/activate
	. ${LANG_ENV_NAME}/bin/activate; pip install tensorflow-gpu sklearn matplotlib h5py scipy

_prod_env: ${PROD_ENV_NAME}/bin/activate
	. ${PROD_ENV_NAME}/bin/activate; pip install tensorflow tensorflowjs

language: _lang_env
	./train.py --train-language-model \
		--conv-length 6 \
		--num-convolutions 512 \
		--num-epochs 100
	./visualize.py

binary: _lang_env
	./train.py
	./visualize.py

productionize: _prod_env
	. ${PROD_ENV_NAME}/bin/activate; ./productionize.py

clean:
	rm -rf outputs/ web/ model.h5
	find -iname "*.pyc" -delete

