import cv2

from prediccion import prediccion

categorias=["Gaviota occidental","Gallinas","Coragyps atratus (Gallinazo)","Tacuarita","Mirlo de alas rojas",
 "green violetear","Cardenal norteño","Hapalopsittaca amazonina","scarlet tanager",
 "momoto amazónico (barranquillo coronado)","Yellow headed Blackbird(Mirlo de cabeza amarilla)","Lazuli Bunting"]
reconocimiento=prediccion()
imagenPrueba=cv2.imread("test/12/12_12_4.jpg",0)
indiceCategoria=reconocimiento.predecir(imagenPrueba)
print("La imagen cargada es ",categorias[indiceCategoria-1])
while True:
    cv2.imshow("imagen",imagenPrueba)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()