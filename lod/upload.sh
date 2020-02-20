#!/bin/sh

# This is a bit messy
if [ "$#" -lt 1 ]; then
        graph="http://example.com/sitc/submission/"
        endpoint="http://fuseki.gsi.upm.es/hotels/data"
else if [ "$#" -lt 2 ]; then
        endpoint=$1
        graph_base="http://example.com/sitc"
     else
         if [ "$#" -lt 3 ]; then
             endpoint=$1
             graph=$2
         else
             echo "Usage: $0 [<endpoint>] [<graph_base_uri>]"
             echo
             exit 1
         fi
     fi
fi


upload(){
    name=$1
    file=$2
    echo '###'
    echo "Uploading: $graph"
    echo "Graph: $graph"
    echo "Endpoint: $endpoint"
    curl -X POST \
         --digest -u admin:$PASSWORD \
         -H Content-Type:text/turtle \
         -T "$file" \
         --data-urlencode graph=$graph_base/$name \
	 -G $endpoint 

}


total=0
echo -n "Password: "
read -s PASSWORD

echo "Uploading synthethic"
upload "synthetic" synthetic/reviews.ttl || exit 1

for i in *.ttl; do
    identifier=$(echo ${i%.ttl} | md5sum | awk '{print $1}')
    echo "Uploading $i"
    upload $identifier $i 
    total=$((total + 1))
done
echo Uploaded $total
