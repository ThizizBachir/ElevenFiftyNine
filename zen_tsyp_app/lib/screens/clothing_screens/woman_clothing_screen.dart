
import 'package:flutter/material.dart';
import 'package:zen_tsyp_app/widgets/clothing_item.dart';


class WomanClothingScreen extends StatefulWidget {
  static const routeName = '/Choose_clothing_screen';
  const WomanClothingScreen({super.key});

  @override
  State<WomanClothingScreen> createState() => _WomanCothingScreenState();
}

class _WomanCothingScreenState extends State<WomanClothingScreen> {
  final List<ClothingItem> clothesList = [
   const  ClothingItem(title: 'Coat',imageAsset: 'assets/woman-1.jpg',price: 50,size: 'L',description: 'Crafted with precision, this shirt combines soft, durable fabric with a tailored fit, perfect for casual or formal wear',),
    const ClothingItem(title:'Coat' ,imageAsset: 'assets/woman-2.jfif', price: 80, size: 'M',description: 'These versatile pants offer a perfect blend of comfort and style, featuring a tailored fit and premium fabric for any occasion.',),
    const ClothingItem(title:'Pants' ,imageAsset: 'assets/woman-3.jfif', price: 80, size: 'L' ,description: 'Experience ultimate comfort and sophistication with this classic shirt, designed to elevate your wardrobe effortlessly',),
    const ClothingItem(title:'Shirt' ,imageAsset: 'assets/woman-4.jfif', price: 80, size: 'M',description: 'Made from high-quality materials, this shirt boasts a modern design with attention to detail, making it a versatile wardrobe staple',),
    const ClothingItem(title:'Dress' ,imageAsset: 'assets/woman-5.jpg', price: 80, size: 'S', description: 'Crafted for durability and elegance, these pants provide a sleek design with a comfortable feel, ideal for work or leisure.',),
    const ClothingItem(title: 'Dress',imageAsset: 'assets/woman-6.jpg',price: 50,size: 'XL',description: 'This coat combines timeless elegance with superior warmth, crafted from premium materials to keep you stylish and comfortable in colder weather',),
    const ClothingItem(title: 'Dress',imageAsset: 'assets/woman-7.jpg',price: 50,size: 'L',description: 'Designed for both function and fashion, this coat features a tailored fit, durable fabric, and thoughtful details for a sophisticated look',),
    const ClothingItem(title: 'Dress',imageAsset: 'assets/woman-8.jfif',price: 50,size: 'L',description: 'A classic coat that blends luxurious texture with modern design, offering versatile wearability for any occasion',),


  ];
  final itemNameController = TextEditingController();
  final itemSize =TextEditingController();
  final itemPrice = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        padding: const EdgeInsets.all(10),
        child:  GridView.builder(
          itemCount: clothesList.length,
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          childAspectRatio: 3/2,
          crossAxisSpacing: 10,
          mainAxisSpacing: 10
        ), itemBuilder: (ctx,i){
          return clothesList[i];
        }),
      )
    );
  }
}