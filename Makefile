FOLDER:=.
ERROR:=255

exec:
	find $(FOLDER) -iname '*.ipynb' -print0 | xargs -n 1 -0 sh -c 'jupyter nbconvert --execute --ClearOutputPreprocessor.enabled=True --inplace $$0 || exit $(ERROR)'

clean:
	find $(FOLDER) -iname '*.ipynb' -print0 | xargs -n 1 -0 sh -c 'nbstripout $$0 || exit $(ERROR)'


.PHONY: exec clean
