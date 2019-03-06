FOLDER:=.

exec:
	find $(FOLDER) -iname '*.ipynb' -print0 | xargs -n 1 -0 sh -c 'jupyter nbconvert --execute --ClearOutputPreprocessor.enabled=True --inplace $$0 || exit 255'

clean:
	find $(FOLDER) -iname '*.ipynb' -print0 | xargs -n 1 -0 sh -c 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $$0 || exit 255'


.PHONY: exec clean
