// script.js

// Supposons que 'clusters' est une liste de clusters de films obtenus après le clustering
// Chaque cluster est une liste d'objets avec les détails des films

// Créer le graphe en utilisant D3.js
// ...

// Par exemple, pour créer des cercles pour chaque film dans le cluster 0 :
const cluster0 = clusters[0];
const node = svg.selectAll(".node")
    .data(cluster0)
    .enter()
    .append("circle")
    .attr("class", "node")
    .attr("r", 10)
    .style("fill", "blue")
    .on("click", function(d) {
        // Gérer l'événement de clic pour afficher les détails du film d
        // Par exemple, vous pouvez afficher les détails dans une section dédiée de la page HTML
    });
// ...

// Vous pouvez également appliquer des styles différents pour chaque cluster ou utiliser des couleurs pour représenter les genres, les années, ou tout autre attribut de vos films.
